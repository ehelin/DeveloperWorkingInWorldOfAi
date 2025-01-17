from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json

# Load the tokenizer and model from Hugging Face
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Global set to store all previously generated responses
previous_responses = set()

def filter_response(response, src_string, add_length, use_rFind):    
    pos = 0
    if use_rFind:
        pos = response.rfind(src_string)
    else: 
        pos = response.find(src_string)

    if pos != -1:
        if add_length:
            response = response[pos + len(src_string):]
        else:    
            response = response[:pos]

    return response

def generate_response(input_line, max_attempts=3):
    try:
        # Dynamically construct the prompt with previous responses
        previous_suggestions_str = ", ".join(previous_responses)
        prompt = f"Previous suggestions: {previous_suggestions_str}." if previous_responses else "Previous suggestions: None."
        promptWithInput = f"{prompt}. {input_line}"
        
        # print("def generate_response(args) - promptWithInput - "+ promptWithInput)

        response = None
        attempts = 0
        
        # print("def main() - before while loop")

        while attempts < max_attempts:
            attempts += 1
            
            # print("def main() - while loop - iteration " + str(attempts))

            # Encode the input text
            inputs = tokenizer(promptWithInput, return_tensors="pt")
            
            # Generate a response with adjusted parameters
            outputs = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,  # Enable sampling for varied responses
                temperature=0.2,  # Add randomness
                top_p=0.9,  # Enable nucleus sampling
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # print("def main() - before if outputs.size(0) > 0:")

            # Decode the response
            if outputs.size(0) > 0:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # print("def main() - raw response " + response)

                # # Filter and clean up the response   
                # response = filter_response(response, promptWithInput, True, False)             
                # response = response.replace(":", "")
                # response = response.replace("\n", "")
                # response = response.replace("Suggest one", "")
                # response = response.replace("Suggestion", "")

                # pos = 0
                # pos = response.find(".")
                # if pos == 0:
                #     response = response[1:]

                # response = filter_response(response, ".", False, False)
                                
                # print("def main() - filtered response " + response)

                # Check if the response is unique
                if response not in previous_responses:
                    break  # Unique response found

        if response is None or response in previous_responses:
            response = "No new unique response could be generated."

        # Store the response in the global set
        previous_responses.add(response)
        
        # print("def main() - leaving - response " + response)

        return response

    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    # print("Python model ready")  # Signal to C# that Python is ready
    sys.stdout.flush()

    while True:
        # Read input from standard input
        input_line = sys.stdin.readline().strip()

        # print("def main() - input_line" + input_line)
        
        # Exit condition
        if input_line == "exit":
            # print("Python exiting")
            sys.stdout.flush()
            break
        
        # Generate response
        response = generate_response(input_line)
        
       # print("def main() - response" + response)

        # Debugging information (optional)
        sys.stderr.write(f"DEBUG: Output generated: {response}\n")
        sys.stderr.flush()
        
        # print(response)
        sys.stdout.flush()
        
        # print("def main() - leaving")

if __name__ == "__main__":
    # print("Starting python script...")

    if sys.stdin.isatty():
        # print("Running in interactive mode. Type 'exit' to quit.")
        while True:
            input_text = input("You: ")
            if input_text.lower() == "exit":
                # print("Exiting interactive mode.")
                break
            
            response = generate_response(input_text)

            # print("Model:", response)
    else:
        # print("Running in main mode.")
        main()
