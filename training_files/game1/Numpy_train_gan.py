"""
Pure NumPy GAN Training Script
================================
No PyTorch needed! Uses only NumPy for training.
This creates trained weights for the text corruption GAN.
"""

import numpy as np
import random
import pickle
from typing import List, Dict, Tuple


class NumpyGANTrainer:
    """GAN Trainer using pure NumPy (no PyTorch!)"""
    
    def __init__(self, embedding_dim=16, hidden_dim=32):
        # Build vocabulary
        self.char_to_idx, self.idx_to_char = self._build_vocab()
        self.vocab_size = len(self.char_to_idx)
        
        # Dimensions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.generator_weights = self._init_generator()
        self.discriminator_weights = self._init_discriminator()
        
        # Corruption patterns to learn
        self.target_patterns = {
            'a': ['@', '4'], 'e': ['3', '€'], 'i': ['1', '!'], 
            'o': ['0', '*'], 's': ['$', '5'], 't': ['7', '+'],
            'A': ['@', '4'], 'E': ['3', '€'], 'I': ['1', '!'],
            'O': ['0', '*'], 'S': ['$', '5'], 'T': ['7', '+']
        }
        
        # Training sentences
        self.sentences = [
            "Machine Learning is Fun",
            "Artificial Intelligence transforms technology",
            "Deep Neural Networks are powerful",
            "Natural Language Processing enables communication",
            "Data Science drives innovation",
            "Computer Vision recognizes patterns",
            "Reinforcement Learning optimizes decisions",
            "Transfer Learning accelerates training"
        ]
        
        # Training hyperparameters
        self.learning_rate = 0.01
        
    def _build_vocab(self):
        """Build character vocabulary"""
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?@#$%&*()-_=+€")
        chars.extend(['<PAD>', '<UNK>'])
        
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        return char_to_idx, idx_to_char
    
    def _init_generator(self):
        """Initialize generator weights"""
        np.random.seed(42)
        
        weights = {
            'embedding': np.random.randn(self.vocab_size, self.embedding_dim) * 0.1,
            'W1': np.random.randn(self.embedding_dim, self.hidden_dim) * 0.1,
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.randn(self.hidden_dim, 1) * 0.1,  # Corruption probability
            'b2': np.zeros(1)
        }
        
        return weights
    
    def _init_discriminator(self):
        """Initialize discriminator weights"""
        np.random.seed(43)
        
        weights = {
            'embedding': np.random.randn(self.vocab_size, self.embedding_dim) * 0.1,
            'W1': np.random.randn(self.embedding_dim, self.hidden_dim) * 0.1,
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.randn(self.hidden_dim, 1) * 0.1,
            'b2': np.zeros(1)
        }
        
        return weights
    
    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def generator_forward(self, char_indices):
        """Forward pass through generator"""
        # Embed characters
        embedded = self.generator_weights['embedding'][char_indices]
        
        # Average pooling over sequence
        pooled = np.mean(embedded, axis=0)
        
        # Hidden layer
        h1 = self.relu(np.dot(pooled, self.generator_weights['W1']) + self.generator_weights['b1'])
        
        # Output: corruption probability
        corrupt_prob = self.sigmoid(np.dot(h1, self.generator_weights['W2']) + self.generator_weights['b2'])
        
        return corrupt_prob[0]
    
    def discriminator_forward(self, char_indices):
        """Forward pass through discriminator"""
        # Embed characters
        embedded = self.discriminator_weights['embedding'][char_indices]
        
        # Average pooling
        pooled = np.mean(embedded, axis=0)
        
        # Hidden layer
        h1 = self.relu(np.dot(pooled, self.discriminator_weights['W1']) + self.discriminator_weights['b1'])
        
        # Output: quality score
        score = self.sigmoid(np.dot(h1, self.discriminator_weights['W2']) + self.discriminator_weights['b2'])
        
        return score[0]
    
    def encode_text(self, text, max_len=50):
        """Encode text to indices"""
        indices = []
        for char in text[:max_len]:
            idx = self.char_to_idx.get(char, self.char_to_idx['<UNK>'])
            indices.append(idx)
        
        # Pad
        while len(indices) < max_len:
            indices.append(self.char_to_idx['<PAD>'])
        
        return np.array(indices)
    
    def create_target_corruption(self, text):
        """Create target corruption"""
        chars = list(text)
        
        for i, char in enumerate(text):
            if char == ' ':
                continue
            
            # Keep first letters mostly
            if i == 0 or (i > 0 and text[i-1] == ' '):
                if random.random() < 0.75:
                    continue
            
            # Apply corruption
            if random.random() < 0.55:
                if char in self.target_patterns:
                    chars[i] = random.choice(self.target_patterns[char])
                elif char.isalpha():
                    if random.random() < 0.4:
                        chars[i] = '_'
        
        return ''.join(chars)
    
    def train_step(self, original_text, corrupted_text):
        """Single training step with simplified gradients"""
        
        # Encode
        orig_indices = self.encode_text(original_text)
        corr_indices = self.encode_text(corrupted_text)
        
        # --- Train Discriminator ---
        # Forward pass on real corruption
        real_score = self.discriminator_forward(corr_indices)
        d_loss_real = (real_score - 0.9) ** 2
        
        # Forward pass on fake (original = bad corruption)
        fake_score = self.discriminator_forward(orig_indices)
        d_loss_fake = (fake_score - 0.1) ** 2
        
        d_loss = d_loss_real + d_loss_fake
        
        # Simple gradient update for discriminator
        # Gradient: move real score towards 0.9, fake score towards 0.1
        d_grad_real = 2 * (real_score - 0.9)
        d_grad_fake = 2 * (fake_score - 0.1)
        
        # Update embeddings slightly
        for idx in corr_indices:
            if idx < self.vocab_size:
                self.discriminator_weights['embedding'][idx] -= self.learning_rate * d_grad_real * 0.01
        
        for idx in orig_indices:
            if idx < self.vocab_size:
                self.discriminator_weights['embedding'][idx] -= self.learning_rate * d_grad_fake * 0.01
        
        # --- Train Generator ---
        # Generator wants to create corruptions that discriminator rates highly
        gen_corrupt_prob = self.generator_forward(orig_indices)
        
        # Create a corruption based on current generator
        generated_corruption = self._apply_generator_corruption(original_text, gen_corrupt_prob)
        gen_corr_indices = self.encode_text(generated_corruption)
        
        # Get discriminator score
        gen_score = self.discriminator_forward(gen_corr_indices)
        
        # Generator loss: want score to be high (close to 0.9)
        g_loss = (gen_score - 0.9) ** 2
        
        # Simple gradient update for generator
        g_grad = 2 * (gen_score - 0.9)
        
        # Update generator embeddings
        for idx in orig_indices:
            if idx < self.vocab_size:
                self.generator_weights['embedding'][idx] -= self.learning_rate * g_grad * 0.01
        
        return d_loss, g_loss
    
    def _apply_generator_corruption(self, text, corruption_prob):
        """Apply corruption based on generator's current probability"""
        chars = list(text)
        
        for i, char in enumerate(text):
            if char == ' ':
                continue
            
            # Keep first letters
            if i == 0 or (i > 0 and text[i-1] == ' '):
                if random.random() < 0.7:
                    continue
            
            # Corrupt based on learned probability
            if random.random() < corruption_prob:
                if char in self.target_patterns:
                    chars[i] = random.choice(self.target_patterns[char])
                elif char.isalpha():
                    if random.random() < 0.4:
                        chars[i] = '_'
        
        return ''.join(chars)
    
    def train(self, epochs=100):
        """Main training loop"""
        print("=" * 70)
        print("  Training NumPy GAN for Text Corruption")
        print("=" * 70)
        print(f"Epochs: {epochs}")
        print(f"Sentences: {len(self.sentences)}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Learning rate: {self.learning_rate}")
        print("=" * 70)
        
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            # Shuffle sentences
            random.shuffle(self.sentences)
            
            # Train on each sentence
            for sentence in self.sentences:
                # Create target corruption
                corrupted = self.create_target_corruption(sentence)
                
                # Train step
                d_loss, g_loss = self.train_step(sentence, corrupted)
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
            
            # Average losses
            avg_d_loss = epoch_d_loss / len(self.sentences)
            avg_g_loss = epoch_g_loss / len(self.sentences)
            
            # Decay learning rate
            if (epoch + 1) % 25 == 0:
                self.learning_rate *= 0.9
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}]")
                print(f"  D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
                print(f"  Learning Rate: {self.learning_rate:.4f}")
                
                # Show example
                test_sentence = random.choice(self.sentences)
                gen_prob = self.generator_forward(self.encode_text(test_sentence))
                generated = self._apply_generator_corruption(test_sentence, gen_prob)
                
                print(f"  Original:  {test_sentence}")
                print(f"  Generated: {generated}")
                print(f"  Gen Prob:  {gen_prob:.3f}")
        
        print("\n" + "=" * 70)
        print("  ✅ Training Complete!")
        print("=" * 70)
    
    def test_corruptions(self):
        """Test the trained model"""
        print("\n" + "=" * 70)
        print("  Testing Trained GAN")
        print("=" * 70)
        
        test_sentences = [
            "Machine Learning is Fun",
            "Data Science drives innovation",
            "Neural Networks are powerful"
        ]
        
        for sentence in test_sentences:
            print(f"\nOriginal: {sentence}")
            
            # Generate corruptions
            gen_prob = self.generator_forward(self.encode_text(sentence))
            print(f"Corruption Probability: {gen_prob:.3f}")
            
            for i in range(3):
                corrupted = self._apply_generator_corruption(sentence, gen_prob)
                print(f"  Variant {i+1}: {corrupted}")
    
    def save_weights(self, path='numpy_gan_weights.pkl'):
        """Save trained weights"""
        data = {
            'generator': self.generator_weights,
            'discriminator': self.discriminator_weights,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'target_patterns': self.target_patterns,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n💾 Saved trained weights to: {path}")
        print(f"   File size: {len(pickle.dumps(data)) / 1024:.1f} KB")


def main():
    """Main training function"""
    print("\n🚀 Starting NumPy GAN Training (No PyTorch needed!)...\n")
    
    # Create trainer
    trainer = NumpyGANTrainer(embedding_dim=16, hidden_dim=32)
    
    # Train
    trainer.train(epochs=100)
    
    # Test
    trainer.test_corruptions()
    
    # Save weights
    trainer.save_weights('numpy_gan_weights.pkl')
    
    print("\n✅ All done! You now have trained GAN weights.")
    print("   Use these weights in your game for realistic text corruption!")


if __name__ == "__main__":
    main()