"""BARE (Base-Refine) Example

This example demonstrates building a two-stage synthetic data generation system
that generates an initial example from a base model and then refines it using
an instruct-tuned model. This takes advantage of the generation diversity of 
base models and the steerability of instruct-tuned models.

This is applied example that showcases:
1. Complex operator composition
2. Custom model registration and usage

To run:
    poetry run python src/ember/examples/bare.py
"""

from typing import ClassVar, Dict, List, Type

# Ember API imports
from ember.api.xcs import jit, vmap
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel, Field
from ember.xcs.engine.execution_options import execution_options

###############################################################################
# Task Prompts
###############################################################################

gsm8k_base_prompt_template = """Here are a few examples of grade school math word problems that require performing a sequence of elementary calculations using basic arithmetic operations. A bright middle school student should be able to solve each problem. The numerical answer is provided at the end of each example after ####.

{examples}

EXAMPLE START
"""

gsm8k_refine_prompt_template = """Improve the given grade school math word problem. Edit the problem or answer to be more similar in style to the examples, and disambiguate as necessary, in addition to correcting any errors. Do not change the theme of the problem. A bright middle school student should be able to solve each problem. Problems require no concepts beyond the level of early Algebra. Note how the numerical answer is provided after #### after each brief reasoning for a question. Provide your edited problem in the following format:
"Question:
[question]
Answer: [answer]"
Provide only the question and answer in the given format. Here are some examples of categories and problems on those categories:
{examples}

Now it's your turn. Here is the question and anwer for you to edit:
Question:
{question}
Answer:
{answer}
Provide only the improved question and answer in the given format. Do not include any commentary or notes. Start your response with the question."""

gsm8k_example_template = "Question:\n{question}\nAnswer:\n{answer}"

gsm8k_examples = [
    {
        "question": """Alice has 20 quarters. She wants to exchange them for nickels and so she goes to the bank. After getting back from the bank, she discovers that 20% of the nickels are iron nickels worth $3 each. What is the total value of her money now?""",
        "answer": "A quarter is worth five nickels because .25 / .05 = <<.25/.05=5>>5 She gets 100 nickels from the bank because 20 x 5 = <<20*5=100>>100 20 of the nickels are iron nickels because 100 x .20 = <<100*.20=20>>20 80 of the nickels are regular because 100 - 20 = <<100-20=80>>80 The iron nickels are worth $60 because 20 x 3 = <<20*3=60>>60 The regular nickels are worth $4 because 80 x .05 = <<80*.05=4>>4 Her money is now worth $64 because 60 + 4 = <<60+4=64>>64 #### 64",
    },
    {
        "question": "A church has 120 members. 40% are adults. The rest are children. How many children more children are there than adults?",
        "answer": "There are 48 adults because 120 x .4 = <<120*.4=48>>48 60% of members are children because 100 - 40 = <<100-40=60>>60 There are 72 children because 120 x .6 = <<120*.6=72>>72 There are 24 more children than adults because 72 - 48 = <<72-48=24>>24 #### 24",
    },
    {
        "question": "Lisa is looking to attempt a World Record. She has decided to try and match Joey Chestnut's record of eating 75 full hotdogs, buns included, in 10 minutes. Halfway through the time Lisa has eaten 20 hotdogs. How many hotdogs will she have to eat per minute to at least tie Joey Chestnut's record?",
        "answer": "Joey Chestnut ate 75 hotdogs to claim the record and Lisa has eaten 20 hot dogs so far, so she still needs to eat 75-20=<<75-20=55>>55 hotdogs to tie Joey Chestnut. Lisa has a 10 minute time period to eat the hotdogs and half the time has already passed, which means Lisa has 10/2=<<10/2=5>>5 minutes left until the competition is over. If she needs to eat 55 hotdogs to tie Joey Chestnut and there are 5 minutes left in the competition period then she needs to eat 55/5=<<55/5=11>>11 hot dogs per minute to have a chance of tying for a win. #### 11",
    },
]

###############################################################################
# Input/Output Models
###############################################################################

class BAREInput(EmberModel):
    """Input for the BARE reasoning system."""
    base_prompt_template: str = Field(description="The prompt template for base model generation. Must include a field for 'examples'")
    instruct_prompt_template: str = Field(description="The prompt template for instruct refinement. Must include a field for 'examples' and a field for each element of the candidate")
    example_template: str = Field(description="The template for each example. Must include a field for each element of an example")
    examples: List[Dict[str, str]] = Field(description="List of examples for base model generation")
    start: str = Field(description="Example start delineator")
    stop: str = Field(description="Example end delineator")

class BaseInput(EmberModel):
    """Input for the base generation step."""
    prompt_template: str = Field(description="The prompt template for base model generation. Must include a field for 'examples'")
    example_template: str = Field(description="The template for each example. Must include a field for each element of an example")
    examples: List[Dict[str, str]] = Field(description="List of examples for base model generation")
    start: str = Field(description="Example start delineator")
    stop: str = Field(description="Example end delineator")

class RefineInput(EmberModel):
    """Input for the instruct refinement step."""
    prompt_template: str = Field(description="The prompt template for instruct refinement. Must include a field for 'examples' and a field for each element of the candidate")
    example_template: str = Field(description="The template for each example. Must include a field for each element of an example")
    examples: List[Dict[str, str]] = Field(description="List of examples for base model generation")
    candidate: Dict[str, str] = Field(description="The candidate response to refine")

class RawOutput(EmberModel):
    """Output for the base generation and instruct refinement steps."""
    response: str = Field(description="The raw generated response")

class ParserOutput(EmberModel):
    """Output for response parser."""
    parsed_output: Dict[str, str] = Field(description="The parsed result")

class BARESpecification(Specification):
    """Specification for the BARE operator."""
    input_model: Type[EmberModel] = BAREInput
    structured_output: Type[EmberModel] = ParserOutput

class BaseGenerationSpecification(Specification):
    """Specification for the BaseGeneration operator."""
    input_model: Type[EmberModel] = BaseInput
    structured_output: Type[EmberModel] = RawOutput

class InstructRefinementSpecification(Specification):
    """Specification for the InstructRefinement operator."""
    input_model: Type[EmberModel] = RefineInput
    structured_output: Type[EmberModel] = RawOutput

class ParserSpecification(Specification):
    """Specification for the ParseResponse operator."""
    input_model: Type[EmberModel] = RawOutput
    structured_output: Type[EmberModel] = ParserOutput


###############################################################################
# BARE Components
###############################################################################

@jit()
class BaseGeneration(Operator[BaseInput, RawOutput]):
    """Operator that generates a candidate from a BaseInput."""

    model_name: str
    temperature: float
    specification: ClassVar[Specification] = BaseGenerationSpecification()

    def __init__(self, *, model_name: str, temperature: float = 1.0) -> None:
        self.model_name = model_name
        self.temperature = temperature

        # Configure internal LM module using the API
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
            )
        )
    
    def format_prompt(self, inputs: BaseInput) -> str:
        prompt = inputs.prompt_template
        examples = inputs.examples
        examples_str = ""
        for example in examples:
            example_str = inputs.example_template.format(**example)
            example_str = f"\n{inputs.start}\n{example_str}\n{inputs.stop}\n"
            examples_str += example_str
        prompt = prompt.format(examples=examples_str)
        prompt += f"\n{inputs.start}"
        return prompt

    def forward(self, *, inputs: BaseInput) -> Dict[str, str]:
        prompt = self.format_prompt(inputs=inputs)
        return RawOutput(response=self.lm_module(prompt=prompt))

class InstructRefinement(Operator[RefineInput, RawOutput]):
    """Operator that generates a candidate from a BaseInput."""

    model_name: str
    temperature: float
    specification: ClassVar[Specification] = InstructRefinementSpecification()

    def __init__(self, *, model_name: str, temperature: float = 1.0) -> None:
        self.model_name = model_name
        self.temperature = temperature

        # Configure internal LM module using the API
        self.lm_module = LMModule(
            config=LMModuleConfig(
                model_name=model_name,
                temperature=temperature,
            )
        )
    
    def format_prompt(self, inputs: RefineInput) -> Dict[str, str]:
        prompt = inputs.prompt_template
        examples = inputs.examples
        examples_str = ""
        for example in examples:
            example_str = inputs.example_template.format(**example)
            example_str = f"\n{example_str}\n"
            examples_str += example_str
        format_dict = {**inputs.candidate, "examples": examples_str}
        prompt = prompt.format(**format_dict)
        return prompt

    def forward(self, *, inputs: RefineInput) -> str:
        prompt = self.format_prompt(inputs=inputs)
        return RawOutput(response=self.lm_module(prompt=prompt))

@jit()
class ParseResponse(Operator[RawOutput, ParserOutput]):
    """Operator that parses a raw response into structured data."""
    specification: ClassVar[Specification] = ParserSpecification()

    def forward(self, *, inputs: RawOutput) -> Dict[str, str]:
        parsed = inputs.response.split("Question:")[1].split("Answer:")
        result = {
            "question": parsed[0].strip(),
            "answer": parsed[1].strip(),
        }
        return ParserOutput(parsed_output=result)

###############################################################################
# Full BARE Operator
###############################################################################
@jit()
class BARE(Operator[BAREInput, ParserOutput]):
    """Operator that performs the BARE reasoning system."""

    base_generation: BaseGeneration
    instruct_refinement: InstructRefinement
    parse_response: ParseResponse
    specification: ClassVar[Specification] = BARESpecification()

    def __init__(self, base_model_name: str, base_temp: float, refine_model_name: str, refine_temp: float) -> None:
        self.base_generation = BaseGeneration(model_name=base_model_name, temperature=base_temp)
        self.instruct_refinement = InstructRefinement(model_name=refine_model_name, temperature=refine_temp)
        self.parse_response = ParseResponse()

    def forward(self, *, inputs: BAREInput) -> Dict[str, str]:
        # Step 1: Base generation
        base_inputs = BaseInput(
            prompt_template=inputs.base_prompt_template,
            example_template=inputs.example_template,
            examples=inputs.examples,
            start=inputs.start,
            stop=inputs.stop,
        )
        base_output = self.base_generation(inputs=base_inputs)

        # Step 2: Parse response
        parsed_base_output = self.parse_response(inputs=base_output)
        # print("=========BASE GENERATION=========")
        # for key, value in parsed_base_output.parsed_output.items():
        #     print(f"{key}: {value}")
        # print("=================================")

        # Step 3: Instruct refinement
        refine_inputs = RefineInput(
            prompt_template=inputs.instruct_prompt_template,
            example_template=inputs.example_template,
            examples=inputs.examples,
            candidate=parsed_base_output.parsed_output,
        )
        refine_output = self.instruct_refinement(inputs=refine_inputs)
        print(refine_output)
        # Step 4: Parse response
        parse_output = self.parse_response(inputs=refine_output)

        return parse_output

def run_bare_in_parallel():
    """Run BARE operator in parallel to generate multiple examples."""
    # Create an instantiation of BARE operator
    bare_op = BARE(
        base_model_name="gpt-4o-mini",
        base_temp=1.0,
        refine_model_name="gpt-4o-mini",
        refine_temp=0.7
    )

    # Create vectorized version for parallel processing
    parallel_bare = vmap(bare_op, in_axes={"seed": 0})

    # Create batch input with multiple seeds
    batch_input = {
        # Same for all runs
        "base_prompt_template": gsm8k_base_prompt_template,
        "instruct_prompt_template": gsm8k_refine_prompt_template,
        "example_template": gsm8k_example_template,
        "examples": gsm8k_examples,
        "start": "EXAMPLE START",
        "stop": "EXAMPLE END",

        # Different for each parallel run. Not used in this example, but could
        # be used to seed each generation differently by modifying Input models
        # and prompt formatting functions.
        "seed": [None] * 3
    }

    # exec in parallel with optimal performance
    results = parallel_bare(inputs=batch_input)

    # Process results
    print(results)
    for i, result in enumerate(results["result"]):
        print(f"Example {i+1}:")
        print(f"  Question: {result.parsed_output['question']}")
        print(f"  Answer: {result.parsed_output['answer']}...")

    return results

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    # Define input for the BARE system
    run_bare_in_parallel()
