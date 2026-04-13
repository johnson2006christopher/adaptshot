import gradio as gr

def launch_demo():
    """
    Skeleton for the Gradio UI.
    To be fully implemented in Day 2.
    """
    def predict(img):
        return {"Placeholder": 0.5}, "Model not loaded yet"

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(), gr.Textbox()],
        title="AdaptShot Demo",
        description="Loading full functionality..."
    )
    # demo.launch() # Commented out to prevent accidental running during setup
    print("UI module ready.")

if __name__ == "__main__":
    launch_demo()
