import { RuleTemplate } from './natural/lib/natural/brill_pos_tagger/index.d';
// src/alerts.ts
//import * as notifier from 'node-notifier';// vaish: error in my pc, "cannot find module 'node-notifier' or its corresponding type declarations"
import * as notifier from '/Program Files/.devrev/devrev-snaps-typescript-template/code/src/Sentiment-Analysis/src/node-notifier';

export const sendAlert = (message: string) => {
    notifier.notify(
        {
          title: 'Test Notification',
          message: 'Click or dismiss this notification',
          wait: true,
        },
        (err: any, response: any, metadata: any) => {
          if (err) {
            console.error('An error occurred:', err);
          } else {
            console.log('Response:', response);
            console.log('Metadata:', metadata);
          }
        }
      );
   
};


  