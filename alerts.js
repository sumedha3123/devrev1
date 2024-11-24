"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.sendAlert = void 0;
// src/alerts.ts
var node_notifier_1 = require("node-notifier");
var sendAlert = function (message) {
    node_notifier_1.default.notify({
        title: 'Test Notification',
        message: 'Click or dismiss this notification',
        wait: true,
    }, function (err, response, metadata) {
        if (err) {
            console.error('An error occurred:', err);
        }
        else {
            console.log('Response:', response);
            console.log('Metadata:', metadata);
        }
    });
};
exports.sendAlert = sendAlert;
