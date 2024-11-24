"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var natural_1 = require("natural");
var natural_2 = require("natural");
// Initialize the sentiment analyzer
var sentiment = new natural_1.SentimentAnalyzer('English', natural_2.PorterStemmer, 'afinn');
// Function to analyze sentiment
var analyzeSentiment = function (text) {
    var score = sentiment.getSentiment(text.split(' '));
    return score;
};
// Example usage
var exampleText = "I am very frustrated with the service.";
var score = analyzeSentiment(exampleText);
console.log("Sentiment score for the text: \"".concat(exampleText, "\" is ").concat(score));
