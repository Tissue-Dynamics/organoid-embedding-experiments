"""Symbolic Aggregate approXimation (SAX) embedding implementation."""

import numpy as np
from typing import Optional, List
import logging
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter

from ..base import FeatureEmbedder

logger = logging.getLogger(__name__)


class SAXEmbedder(FeatureEmbedder):
    """
    Symbolic Aggregate approXimation (SAX) embedding for time series data.
    
    Converts time series to symbolic representations and extracts features
    from symbol patterns, frequencies, and transitions.
    """
    
    def __init__(self, n_segments: int = 20, n_symbols: int = 4, 
                 alphabet: Optional[List[str]] = None,
                 strategy: str = 'uniform', window_size: Optional[int] = None,
                 word_length: int = 3, include_transitions: bool = True,
                 vectorizer_type: str = 'count', **kwargs):
        """
        Initialize SAX embedder.
        
        Args:
            n_segments: Number of segments to divide time series into
            n_symbols: Size of alphabet for discretization
            alphabet: Custom alphabet (if None, uses letters)
            strategy: Binning strategy ('uniform', 'quantile', 'normal')
            window_size: Sliding window size (if None, uses n_segments)
            word_length: Length of SAX words to extract
            include_transitions: Whether to include symbol transition features
            vectorizer_type: Type of vectorizer ('count', 'tfidf')
        """
        super().__init__(name="SAX", **kwargs)
        self.n_segments = n_segments
        self.n_symbols = n_symbols
        self.alphabet = alphabet or [chr(ord('a') + i) for i in range(n_symbols)]
        self.strategy = strategy
        self.window_size = window_size or n_segments
        self.word_length = word_length
        self.include_transitions = include_transitions
        self.vectorizer_type = vectorizer_type
        
        # Initialize SAX transformer
        self.sax = SymbolicAggregateApproximation(
            n_bins=n_symbols,
            strategy=strategy,
            alphabet=self.alphabet
        )
        
        self.vectorizer = None
        
    def _create_sax_words(self, sax_string: str) -> List[str]:
        """
        Create SAX words of specified length from SAX string.
        
        Args:
            sax_string: SAX representation of time series
            
        Returns:
            List of SAX words
        """
        if len(sax_string) < self.word_length:
            return [sax_string]
        
        words = []
        for i in range(len(sax_string) - self.word_length + 1):
            words.append(sax_string[i:i + self.word_length])
        
        return words
    
    def _compute_transition_features(self, sax_string: str) -> np.ndarray:
        """
        Compute symbol transition features.
        
        Args:
            sax_string: SAX representation
            
        Returns:
            Transition features
        """
        if len(sax_string) < 2:
            return np.zeros(self.n_symbols * self.n_symbols)
        
        # Create transition matrix
        transition_counts = np.zeros((self.n_symbols, self.n_symbols))
        
        for i in range(len(sax_string) - 1):
            from_symbol = self.alphabet.index(sax_string[i])
            to_symbol = self.alphabet.index(sax_string[i + 1])
            transition_counts[from_symbol, to_symbol] += 1
        
        # Normalize by total transitions
        total_transitions = np.sum(transition_counts)
        if total_transitions > 0:
            transition_probs = transition_counts / total_transitions
        else:
            transition_probs = transition_counts
        
        return transition_probs.flatten()
    
    def _compute_symbol_features(self, sax_string: str) -> np.ndarray:
        """
        Compute basic symbol frequency features.
        
        Args:
            sax_string: SAX representation
            
        Returns:
            Symbol frequency features
        """
        symbol_counts = Counter(sax_string)
        total_symbols = len(sax_string)
        
        features = []
        for symbol in self.alphabet:
            count = symbol_counts.get(symbol, 0)
            frequency = count / total_symbols if total_symbols > 0 else 0
            features.append(frequency)
        
        # Additional statistical features
        if len(sax_string) > 0:
            # Symbol diversity (entropy)
            probs = np.array([symbol_counts.get(s, 0) for s in self.alphabet]) / total_symbols
            probs = probs[probs > 0]  # Remove zeros for log
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
            
            # Run length statistics
            run_lengths = []
            current_symbol = sax_string[0]
            current_run = 1
            
            for i in range(1, len(sax_string)):
                if sax_string[i] == current_symbol:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_symbol = sax_string[i]
                    current_run = 1
            run_lengths.append(current_run)
            
            mean_run_length = np.mean(run_lengths)
            std_run_length = np.std(run_lengths)
            max_run_length = np.max(run_lengths)
            
            features.extend([entropy, mean_run_length, std_run_length, max_run_length])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract SAX features from time series data.
        
        Args:
            X: Time series data of shape (n_samples, n_timepoints)
            
        Returns:
            SAX features
        """
        n_samples = len(X)
        logger.info(f"Extracting SAX features for {n_samples} time series")
        
        # Convert to SAX representations
        sax_strings = []
        for i in range(n_samples):
            ts = X[i:i+1]  # Keep 2D shape for pyts
            try:
                sax_repr = self.sax.transform(ts)[0]  # Get first (and only) result
                sax_strings.append(''.join(sax_repr))
            except Exception as e:
                logger.warning(f"Failed to convert time series {i} to SAX: {e}")
                sax_strings.append('a' * self.n_segments)  # Fallback
        
        # Extract SAX words for bag-of-words features
        all_words = []
        for sax_string in sax_strings:
            words = self._create_sax_words(sax_string)
            all_words.append(' '.join(words))
        
        # Fit vectorizer if not already fitted
        if self.vectorizer is None:
            if self.vectorizer_type == 'count':
                self.vectorizer = CountVectorizer(
                    ngram_range=(1, 1),  # Single words
                    max_features=1000,   # Limit vocabulary size
                    binary=False
                )
            elif self.vectorizer_type == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 1),
                    max_features=1000,
                    binary=False
                )
            else:
                raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")
            
            bag_of_words_features = self.vectorizer.fit_transform(all_words).toarray()
        else:
            bag_of_words_features = self.vectorizer.transform(all_words).toarray()
        
        # Compute additional features
        symbol_features = np.array([self._compute_symbol_features(s) for s in sax_strings])
        
        features = [bag_of_words_features, symbol_features]
        
        if self.include_transitions:
            transition_features = np.array([self._compute_transition_features(s) for s in sax_strings])
            features.append(transition_features)
        
        # Concatenate all features
        return np.hstack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        names = []
        
        # Bag-of-words features
        if self.vectorizer is not None:
            bow_names = [f"sax_word_{word}" for word in self.vectorizer.get_feature_names_out()]
            names.extend(bow_names)
        
        # Symbol frequency features
        for symbol in self.alphabet:
            names.append(f"symbol_freq_{symbol}")
        names.extend(['symbol_entropy', 'mean_run_length', 'std_run_length', 'max_run_length'])
        
        # Transition features
        if self.include_transitions:
            for i, from_symbol in enumerate(self.alphabet):
                for j, to_symbol in enumerate(self.alphabet):
                    names.append(f"transition_{from_symbol}_to_{to_symbol}")
        
        return names
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SAXEmbedder':
        """
        Fit SAX embedder.
        
        Args:
            X: Training time series data
            y: Ignored
            
        Returns:
            Self for method chaining
        """
        # Fit the SAX transformer
        self.sax.fit(X)
        
        # Extract features to fit the vectorizer
        features = self.extract_features(X)
        
        # Fit scaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        
        self.embedding_dim = features.shape[1]
        self.is_fitted = True
        
        logger.info(f"SAX embedder fitted with {len(X)} samples, {self.embedding_dim} features")
        return self