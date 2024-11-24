// src/thresholds.ts
export interface FrustrationThresholds {
    sentimentScore: number;
    keywords: string[];
}

export const thresholds: FrustrationThresholds = {
    sentimentScore: -1, // Example threshold for negative sentiment
    keywords: ['angry', 'frustrated', 'hate', 'disappointed'],
};