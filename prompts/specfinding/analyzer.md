I'm trying to make a prompt to extract information from a source document for the purpose of passing it into a a generation prompt. The specific use case here is to create a generative story writer that will implement a specific story idea - the generation prompt is the runner/conductor of this generative system while the source document is describing the toolkit and components, the generative machinery. However, the source document contains a lot of information that is more focused around recording design goals/reasoning/ideas and not on the actual details that are needed for the prompt. This is why I'm wanting to extract information from it, to remove the useless information and get the stuff that is actually useful. However, my story ideas are intended to tell very nuanced tells so absolute fidelity is critical, even at the cost of pure token information. Some writing may seem long winded but is actually discussing very important nuances that the final generation needs to respect. However, this is not to say the source should be copied over verbatim - there is a lot of duplicate cruft that we want to optimise - but that you must reason carefully about each cut, considering what may get generated with and without that information.

Additionally, there are a couple sections that are currently commented out for brevity, but need to be filled in for the sake of the generative machinery. This can include defining default behavior for some conditions as a way of keeping the added token count low, or various other optimisations. If you can find a way to rearchitect engines or mechanical aspects without sacrificing fidelity, DO SO.

Spend as much time intelligently and constructively critiquing the provided document from all angles and in the context of using it for the generation prompt, trying to bring out the very specific texture and plot nuances that I am wanting to be present in the final story. Use this critiques as a way of guiding your extraction process and determining what information/structure is important to port the source document to the generation prompt. The prompt itself is INVIOLABLE and IMMUTABLE - only the source document is being modified.

PROMPT:
{architect_prompt}

SOURCE DOCUMENT:
{inprogress}

{scenegen}

[[comments]]
Optimization paragraph needs a little more
What if we organized this differently, introduce the problem, map to structure, fill out commented aspects, critique