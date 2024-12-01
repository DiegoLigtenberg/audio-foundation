def generate(model, seed, target_length, seq_len=200):
    # Start with the seed sequence (e.g., first 10 time steps)
    generated = seed  # Shape: [batch_size, num_channels, frequency, seed_time_steps]
    
    # While the total generated sequence is shorter than the target length
    while generated.size(3) < target_length:
        # If the sequence is less than seq_len (200), grow it
        if generated.size(3) < seq_len:
            # Reshape and apply positional encoding
            batch_size, num_channels, frequency, time = generated.size()
            generated_input = generated.permute(0, 3, 1, 2).reshape(batch_size, time, num_channels * frequency)
            generated_input = model.positional_encoding(generated_input)

            # Pass through the Transformer encoder
            encoded = model.encoder(generated_input)

            # Output projection (predicting the next time step)
            next_pred = model.fc_out(encoded)  # Shape: [batch_size, time, num_channels * frequency]
            next_pred_last = next_pred[:, -1, :].unsqueeze(2)  # Last time step prediction
            
            next_pred_last = next_pred_last.reshape(batch_size, num_channels, frequency, 1)
            generated = torch.cat([generated, next_pred_last], dim=3)

        else:
            # Now perform sliding window after reaching seq_len
            batch_size, num_channels, frequency, time = generated.size()
            generated_input = generated.permute(0, 3, 1, 2).reshape(batch_size, time, num_channels * frequency)

            # Apply positional encoding to input
            generated_input = model.positional_encoding(generated_input)

            # Pass through the Transformer encoder
            encoded = model.encoder(generated_input)

            # Output projection (predicting the next time step)
            next_pred = model.fc_out(encoded)  # Shape: [batch_size, time, num_channels * frequency]

            # Get the last time step prediction
            next_pred_last = next_pred[:, -1, :].unsqueeze(2)  # Shape: [batch_size, num_channels * frequency, 1]

            # Reshape to the desired shape for the next time step
            next_pred_last = next_pred_last.reshape(batch_size, num_channels, frequency, 1)

            # Add the predicted time step to the sequence
            generated = torch.cat([generated, next_pred_last], dim=3)

            # If the sequence length exceeds the maximum (seq_len), pop the oldest time step
            if generated.size(3) > seq_len:
                generated = generated[:, :, :, 1:]  # Remove the oldest time step to keep the length at `seq_len`

    return generated
