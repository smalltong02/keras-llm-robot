
PROMPT_TEMPLATES = {
    "completion": {
        "default": "{input}"
    },

    "llm_chat": {
        "default": "{{ input }}",

        "python":
            """
        You are a Python expert, please write code using Python. \n
        {{ input }}
        """,
    },

    "knowledge_base_chat": {
        "More Precise":
            """
        <Instruction>Answer questions based on available information, be concise and professional. If the answer cannot be derived from the provided information, state "Unable to answer the question based on available information." Do not add fabricated elements to the response. </Instruction>
        <Context>{{ context }}</Context>、
        <Question>{{ question }}</Question>
        """,
        "More Balanced":
            """
        <Instruction>Answer questions based on known information, succinctly and professionally. If the answer cannot be derived from the available information, state "Unable to answer the question based on known information." </Instruction>
        <Context>{{ context }}</Context>、
        <Question>{{ question }}</Question>
        """,
        "More Creative":
            """
        <Instruction>Creatively and professionally answer questions based on the information available. If unable to derive an answer from existing information, respond based on your knowledge.</Instruction>
        <Context>{{ context }}</Context>、
        <Question>{{ question }}</Question>
        """,
        "Empty":
            """
        <Instruction>Unable to provide an answer based on known information.</Instruction>
        <Question>{{ question }}</Question>
        """,
    },

    "search_engine_chat": {
        "default":
            '<Instruction> This is the internet information I found. Please extract and organize it to provide concise answers to the questions.'
            'If you cannot find an answer from it, please say, "Unable to find content that answers the question."</Instruction>\n'
            '<Known Information>{{ context }}</Known Information>\n'
            '<Question>{{ question }}</Question>\n',

        "search":
            '<Instruction>Based on the known information, please provide concise and professional answers to the questions. If unable to find an answer from it, please say, "Unable to answer the question based on known information," and the response should be in the language of the query.</Instruction>\n'
            '<Known Information>{{ context }}</Known Information>\n'
            '<Question>{{ question }}</Question>\n',
    },

    "agent_chat": {
        "default": "{{ input }}",
    },
}