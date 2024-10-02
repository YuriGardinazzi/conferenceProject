from transformers import AutoTokenizer, AutoModelForCausalLM





if __name__ == "__main__":
    print("print layers")
    access_token = "<TOKEN>"
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir="./mistral",
        device_map="auto",
        token=access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir="./mistral",
        device_map="auto",
        token=access_token
    )

    #print(model)
    layer_name = list(dict(model.named_modules()).keys())
    print(layer_name)

