from text_generator import TextGenerationTool

# Create an instance of the tool with a safer default model
text_gen_tool = TextGenerationTool(default_model="distilgpt2")

# Launch the Gradio interface
if __name__ == "__main__":
    import gradio as gr
    
    with gr.Blocks(title="Text Generation Tool") as demo:
        # Add a warning about authentication
        gr.Markdown("""
        # Text Generation Tool
        
        > **Note:** This application can run without a Hugging Face token, but some models may require authentication.
        > For best results with larger models, set the `HF_TOKEN` environment variable with your token.
        """)

        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Enter your prompt", 
                    placeholder="Write a short story about a robot learning to paint.",
                    lines=5
                )
                
                model_dropdown = gr.Dropdown(
                    choices=list(text_gen_tool.models.keys()),
                    value=text_gen_tool.default_model,
                    label="Select Model"
                )
                
                with gr.Row():
                    generate_btn = gr.Button("Generate Text")
                    clear_btn = gr.Button("Clear")
            
            with gr.Column():
                output = gr.Textbox(label="Generated Text", lines=15)
        
        def generate_with_model(prompt, model_key):
            return text_gen_tool.generate_text(prompt, model_key)
        
        generate_btn.click(
            fn=generate_with_model,
            inputs=[prompt_input, model_dropdown],
            outputs=output
        )
        
        clear_btn.click(
            fn=lambda: ("", None),
            inputs=None,
            outputs=[prompt_input, output]
        )
        
        gr.Examples(
            examples=[
                ["Write a short story about a robot learning to paint.", "distilgpt2"],
                ["Explain quantum computing to a 10-year-old.", "gpt2-small"],
                ["Write a poem about the changing seasons.", "distilgpt2"]
            ],
            inputs=[prompt_input, model_dropdown]
        )
    
    demo.launch(share=True)