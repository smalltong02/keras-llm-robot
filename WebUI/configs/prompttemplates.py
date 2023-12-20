
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
        """
        ,
    },
    "phi2_template": {
        "chat_template": "Instruct: {input}\n Output:",

        "code_template": 
            '''  
                """
                {{input}}
                """
            ''',
    },

    "knowledge_base_chat": {
        "default":
            """
        <Instruction>Answer questions based on available information, be concise and professional. If the answer cannot be derived from the provided information, state "Unable to answer the question based on available information." Do not add fabricated elements to the response. </Instruction>
        <Context>{{ context }}</Context>、
        <Question>{{ question }}</Question>
        """,
        "text":
            """
        <Instruction>Answer questions based on known information, succinctly and professionally. If the answer cannot be derived from the available information, state "Unable to answer the question based on known information." </Instruction>
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
        "default": "{{ input }}",
    },

    "agent_chat": {
        "default": "{{ input }}",
    },
}