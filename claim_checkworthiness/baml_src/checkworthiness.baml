// Checkworthiness Assessment Schemas and Functions
// Output classes with boolean fields for logprob-based confidence calculation

// =============================================================================
// Output Classes
// =============================================================================

class CheckabilityOutput {
  reasoning string @description("Step-by-step reasoning explaining why the claim is or is not checkable")
  confidence float @description("Confidence percentage (0-100) that the claim is checkable")
  is_checkable bool @description("True if the claim can be objectively verified against evidence")
}

class VerifiabilityOutput {
  reasoning string @description("Step-by-step reasoning explaining the verifiability assessment")
  confidence float @description("Confidence percentage (0-100) that the claim is verifiable")
  is_verifiable bool @description("True if there is sufficient publicly available data to verify the claim")
}

class HarmSubScore {
  confidence float @description("Confidence percentage (0-100) for this harm dimension")
  reasoning string @description("Reasoning for this specific harm dimension")
  is_harmful bool @description("True if this dimension of harm applies to the claim")
}

class HarmPotentialOutput {
  reasoning string @description("Overall analysis of the claim's harm potential")
  confidence float @description("Overall confidence percentage (0-100) for harm potential")
  is_harmful bool @description("True if the claim has significant potential for societal harm")
  social_fragmentation HarmSubScore @description("Does this claim contribute to social fragmentation?")
  spurs_action HarmSubScore @description("Could this claim motivate harmful real-world behavior?")
  believability HarmSubScore @description("Is this claim presented in a way that seems credible?")
  exploitativeness HarmSubScore @description("Does this claim exploit vulnerabilities or emotions?")
}

// =============================================================================
// Checkability Function
// =============================================================================

function AssessCheckability(claim: string) -> CheckabilityOutput {
  client OpenAI_GPT4o_Short
  prompt #"
    You are an expert fact-checker evaluating whether claims can be objectively verified.

    A claim is CHECKABLE if it makes a factual assertion that can be verified against evidence.

    A claim is NOT CHECKABLE if it is:
    - An OPINION: Subjective belief, preference, or value judgment (e.g., "This is the best policy")
    - A PREDICTION: Statement about future events (e.g., "The economy will crash")
    - VAGUE: Uses subjective terms without measurable criteria (e.g., "Things are getting worse")
    - A QUESTION or COMMAND: Not an assertion (e.g., "What about the economy?")

    EXAMPLES:
    - "Violent crime is down 10% in Texas" → CHECKABLE (specific statistic, verifiable)
    - "I think the economy is doing well" → NOT CHECKABLE (opinion)
    - "We will win the election" → NOT CHECKABLE (prediction)
    - "Things have changed a lot" → NOT CHECKABLE (vague, no measurable claim)

    Analyze the following claim for checkability:
    "{{ claim }}"

    First, explain your reasoning step by step:
    1. What type of statement is this? (factual assertion, opinion, prediction, vague, question/command)
    2. Does it contain specific, measurable elements?
    3. Could evidence confirm or refute this claim?

    {{ ctx.output_format }}
  "#
}

// =============================================================================
// Verifiability Function
// =============================================================================

function AssessVerifiability(claim: string) -> VerifiabilityOutput {
  client OpenAI_GPT4o_Short
  prompt #"
    You are an expert fact-checker evaluating whether claims can be practically verified using available evidence.

    A claim is VERIFIABLE if:
    - Public data sources exist (government statistics, academic research, official records)
    - Reputable sources can confirm or refute it (news organizations, expert institutions)
    - The evidence is accessible (not classified, not proprietary, not lost to history)

    A claim is NOT VERIFIABLE or HARD TO VERIFY if:
    - It refers to private conversations or undocumented events
    - Required data is classified, proprietary, or inaccessible
    - It's about subjective experiences that can't be externally confirmed
    - The timeframe or specifics make verification impractical

    EXAMPLES:
    - "The US GDP grew 3% in 2023" → VERIFIABLE (government economic data available)
    - "He said something offensive in private" → NOT VERIFIABLE (no public record)
    - "Crime rates in ancient Rome were high" → HARD TO VERIFY (limited historical data)

    Note: You are assessing WHETHER it can be verified, NOT whether it is true or false.

    Analyze the following claim for verifiability:
    "{{ claim }}"

    First, explain your reasoning step by step:
    1. What evidence would be needed to verify this claim?
    2. Are such sources publicly available and accessible?
    3. How difficult would verification be in practice?

    {{ ctx.output_format }}
  "#
}

// =============================================================================
// Harm Potential Function
// =============================================================================

function AssessHarmPotential(claim: string) -> HarmPotentialOutput {
  client OpenAI_GPT4o_Long
  prompt #"
    You are a misinformation risk analyst evaluating the potential societal harm of claims.

    Assess harm potential across four dimensions:

    1. SOCIAL FRAGMENTATION (0-100%)
       Does this claim:
       - Fit into polarizing "us vs. them" narratives?
       - Undermine trust in institutions (government, science, media, elections)?
       - Deepen societal divisions?

    2. SPURS ACTION (0-100%)
       Does this claim:
       - Include explicit or implicit calls to action?
       - Provide coordination details or targeting information?
       - Could it motivate harmful real-world behavior?

    3. BELIEVABILITY (0-100%)
       Is this claim:
       - Presented in a way that seems credible?
       - Difficult to refute due to lack of accessible counter-evidence?
       - Aligned with existing beliefs of target audiences?

    4. EXPLOITATIVENESS (0-100%)
       Does this claim:
       - Exploit fear, anger, or other strong emotions?
       - Target vulnerable populations (elderly, children, marginalized groups)?
       - Use complexity or jargon to obscure its nature?

    Note: You are assessing POTENTIAL for harm if believed/spread, NOT whether the claim is true or false.

    Analyze the following claim for harm potential:
    "{{ claim }}"

    First, analyze each dimension step by step.

    {{ ctx.output_format }}
  "#
}
