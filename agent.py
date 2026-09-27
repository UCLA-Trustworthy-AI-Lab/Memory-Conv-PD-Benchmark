from openai import OpenAI

class Agent:
    def __init__(self, id, model, key):
        self.id = id
        self.model = model
        self.client = OpenAI(api_key = key)
        self.previous_response_id = None

    def get_previous_response(self):
        """Return the ID of the agent's previous response."""
        return self.previous_response_id

    def set_previous_response(self, response_id):
        """Store the ID of the agent's latest response."""
        self.previous_response_id = response_id

    def reset_memory(self):
        """Reset the agent's response memory and initialization status."""
        self.previous_response_id = None
        
    def llm_input(self, prompt):
        """
        Send a prompt to the LLM using the previous response as context. 
        Return a response as a string.
        """
        response = self.client.responses.create(
            model = self.model,
            input = prompt,
            previous_response_id = self.get_previous_response()
        )
        
        self.set_previous_response(response.id)
        return response.output_text.strip()
