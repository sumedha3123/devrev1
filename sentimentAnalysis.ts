import { SentimentAnalyzer } from 'natural';
import { PorterStemmer } from 'natural';

// Initialize the sentiment analyzer
const sentiment = new SentimentAnalyzer('English', PorterStemmer, 'afinn');

// Function to analyze sentiment
const analyzeSentiment = (text: string): number => {
    const score = sentiment.getSentiment(text.split(' '));
    return score;
};

// Example usage
const exampleText = "I am very frustrated with the service.";
const score = analyzeSentiment(exampleText);
console.log(`Sentiment score for the text: "${exampleText}" is ${score}`);