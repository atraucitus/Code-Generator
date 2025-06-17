import torch

SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

def predict(model, sentence, vocab, tokenizer, device, max_len=50):
    model.eval()
    
    # Tokenize the sentence and add SOS and EOS tokens
    tokens = [SOS] + tokenizer(sentence) + [EOS]
    
    # Direct access to token-to-index (stoi) and index-to-token (itos)
    stoi = vocab.get_stoi()  # Token to index mapping
    itos = vocab.get_itos()  # Index to token mapping
    
    # Handle missing tokens by providing the index of <unk>
    input_tensor = [stoi.get(token, stoi[UNK]) for token in tokens]
    
    # Convert to tensor
    input_tensor = torch.tensor(input_tensor, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Pass through the encoder
        hidden, cell = model.encoder(input_tensor)
        
        # Start generating the output code
        input_token = torch.tensor([stoi[SOS]], dtype=torch.long).to(device)
        
        output_tokens = []
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            top1 = output.argmax(1)  # Get the most probable token
            
            # Stop if EOS is encountered
            if top1.item() == stoi[EOS]:
                break
            output_tokens.append(top1.item())
            input_token = top1
        
        # Convert output tokens back to text using itos
        decoded_tokens = [itos[idx] for idx in output_tokens]
        
        # Join tokens into a final string
        return ' '.join(decoded_tokens)
