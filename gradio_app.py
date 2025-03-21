import gradio as gr
from Theme_Classifier import themeClassifier
import pandas as pd
import matplotlib.pyplot as plt

def get_themes(theme_list_str, subtitles_path, save_path):
    print("Button clicked!")  # Check if the function is being called
    theme_list = theme_list_str.split(",")
    print(f"Themes: {theme_list}, Subtitles Path: {subtitles_path}, Save Path: {save_path}")
    
    try:
        theme_classifier = themeClassifier(theme_list)
        output_df = theme_classifier.get_themes(subtitles_path, save_path)
        
        # Sum the themes and create a new DataFrame
        output_df = output_df[theme_list]
        output_df = output_df.sum().reset_index()
        output_df.columns = ['Themes', 'Score']
        print(output_df)  # Debugging output
        
        # Plot using Matplotlib
        fig, ax = plt.subplots()
        ax.bar(output_df['Themes'], output_df['Score'])
        ax.set_title("Harry Potter Themes")
        ax.set_xlabel("Themes")
        ax.set_ylabel("Score")
        
        # Return the Matplotlib figure to Gradio
        return fig 

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1> Theme Classification using Zero Shot Classifier </h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.Plot()  # Placeholder for the plot
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        
                        # Trigger the get_themes function when the button is clicked
                        get_themes_button.click(
                            get_themes, 
                            inputs=[theme_list, subtitles_path, save_path], 
                            outputs=plot  # The output is the Plot object
                        )
    iface.launch(share=True)

if __name__ == "__main__":
    main()
