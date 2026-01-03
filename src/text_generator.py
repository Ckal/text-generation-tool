import os
import requests
import gradio as gr
from transformers import pipeline
from smolagents import Tool

class TextGenerationTool(Tool):
    name = "text_generator"
    description = "This is a tool for text generation. It takes a prompt as input and returns the generated text."
    
    inputs = {
        "text": {
            "type": "string",
            "description": "The prompt for text generation"
        }
    }
    output_type = "string"
    
    # Available text generation models
    models = {
        "distilgpt2": "distilgpt2",  # Smaller model, may work without auth
        "gpt2-small": "sshleifer/tiny-gpt2",  # Tiny model for testing
        "opt-125m": "facebook/opt-125m",  # Small, open model
        "bloom-560m": "bigscience/bloom-560m",
        "gpt2": "gpt2"  # Original GPT-2
    }
    
    def __init__(self, default_model="distilgpt2", use_api=False):
        """Initialize with a default model and API preference."""
        super().__init__()
        self.default_model = default_model
        self.use_api = use_api
        self._pipelines = {}
        
        # Check for API token
        self.token = os.environ.get('HF_TOKEN') or os.environ.get('HF_token')
        if self.token is None:
            print("Warning: No Hugging Face token found. Set HF_TOKEN environment variable for authenticated requests.")
    
    def forward(self, text: str):
        """Process the input prompt and generate text."""
        return self.generate_text(text)
    
    def generate_text(self, prompt, model_key=None, max_length=500, temperature=0.7):
        """Generate text based on the prompt using the specified or default model."""
        # Determine which model to use
        model_key = model_key or self.default_model
        model_name = self.models.get(model_key, self.models[self.default_model])
        
        # Generate using API if specified
        if self.use_api and model_key == "openchat":
            return self._generate_via_api(prompt, model_name)
        
        # Otherwise use local pipeline
        return self._generate_via_pipeline(prompt, model_name, max_length, temperature)
    
    def _generate_via_pipeline(self, prompt, model_name, max_length, temperature):
        """Generate text using a local pipeline."""
        try:
            # Get or create the pipeline
            if model_name not in self._pipelines:
                # Use token if available, otherwise try without it
                try:
                    kwargs = {"token": self.token} if self.token else {}
                    self._pipelines[model_name] = pipeline(
                        "text-generation", 
                        model=model_name, 
                        **kwargs
                    )
                except Exception as e:
                    print(f"Error loading model {model_name}: {str(e)}")
                    # Fall back to tiny-distilgpt2 if available
                    if model_name != "sshleifer/tiny-gpt2":
                        print("Falling back to tiny-gpt2 model...")
                        return self._generate_via_pipeline(prompt, "sshleifer/tiny-gpt2", max_length, temperature)
                    else:
                        raise e
            
            generator = self._pipelines[model_name]
            
            # Generate text
            result = generator(
                prompt, 
                max_length=max_length, 
                num_return_sequences=1, 
                temperature=temperature
            )
        
            # Extract and return the generated text
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
                return result[0]
            
            return str(result)
        except Exception as e:
            return f"Error generating text: {str(e)}\n\nPlease try a different model or prompt."
    
    def _generate_via_api(self, prompt, model_name):
        """Generate text by calling the Hugging Face API."""
        if not self.token:
            return "Error: HF_token not set. Cannot use API."
        
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"inputs": prompt}
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text']
            
            # Fall back to returning the raw response
            return str(result)
            
        except Exception as e:
            return f"Error generating text: {str(e)}"

# For standalone testing
if __name__ == "__main__":
    # Create an instance of the TextGenerationTool
    text_generator = TextGenerationTool(default_model="gpt2")
    
    # Test with a simple prompt
    test_prompt = "Once upon a time in a digital world,"
    result = text_generator(test_prompt)
    
    print(f"Prompt: {test_prompt}")
    print(f"Generated text:\n{result}")