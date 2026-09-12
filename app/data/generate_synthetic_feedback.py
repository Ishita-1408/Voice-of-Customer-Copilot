"""Synthetic customer evidence generation pipeline for VoC Copilot.

Generates 150 deterministic, template-based synthetic customer feedback records
(75 Support Tickets, 40 Interviews, 35 Surveys) across 8 controlled product intelligence
scenarios to validate the downstream VoC pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMNS = [
    "feedback_id",
    "source_type",
    "source_name",
    "feedback_text",
    "created_at",
    "customer_segment",
    "product_area",
    "sentiment",
    "severity",
    "language",
    "data_type",
    "metadata_origin",
    "rating",
    "product_name",
    "product_price",
]

METADATA_ORIGIN_SYNTHETIC = json.dumps({
    "source_type": "synthetic_generator",
    "source_name": "synthetic_generator",
    "feedback_text": "synthetic_template",
    "sentiment": "synthetic_label",
    "rating": "synthetic_scale",
    "product_name": "synthetic_scenario",
    "product_price": "synthetic_scenario",
    "data_type": "synthetic",
    "product_area": "synthetic_label",
    "severity": "synthetic_rule",
    "language": "synthetic_label",
    "customer_segment": "synthetic_cohort",
    "created_at": "synthetic_timestamp",
})

# ==============================================================================
# DETERMINISTIC DATA TEMPLATES & SCENARIO DEFINITIONS
# ==============================================================================

# 1. PAYMENT & CHECKOUT RELIABILITY (35 records: 20 SUP, 8 INT, 7 SUR)
# Increasing trend: Jun(3), Jul(6), Aug(11), Sep(15)
# Disproportionately New User - Mobile
PAYMENT_TEMPLATES = [
    # Support Tickets (20)
    {"text": "My payment failed twice at the final checkout step using UPI, but money was deducted from my account.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-06-12", "prod": "Mobile Checkout Gateway", "price": 1499},
    {"text": "The payment gateway kept spinning on my iPhone and timed out before showing any confirmation.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-06-25", "prod": "Mobile Checkout Gateway", "price": 2899},
    {"text": "I tried paying with credit card and it asked for OTP, but the page crashed right after submitting.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-04", "prod": "Payment Processing", "price": 4500},
    {"text": "The checkout page gave an unknown error code ERR_PAY_TIMEOUT when I tried placing my order.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-07-15", "prod": "Mobile Checkout Gateway", "price": 899},
    {"text": "Payment status remained in 'Processing' for 4 hours. I don't know whether to place the order again.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-07-22", "prod": "Order Checkout API", "price": 3200},
    {"text": "My bank sent an SMS that ₹2,100 was debited, but the app shows 'Payment Incomplete'. Please resolve immediately.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-29", "prod": "Payment Processing", "price": 2100},
    {"text": "Every time I tap 'Pay Now' on my Android phone, the payment screen goes blank and returns to the cart.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-08-02", "prod": "Mobile App Checkout", "price": 1250},
    {"text": "Card payment got declined repeatedly without telling me why. I had to use a different platform.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Desktop", "date": "2026-08-07", "prod": "Web Checkout Gateway", "price": 5400},
    {"text": "I needed three attempts to complete my UPI transaction because the QR code expired prematurely.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-11", "prod": "UPI Gateway Service", "price": 750},
    {"text": "The app crashed at the payment verification screen. Now I have two pending charges on my card.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-08-16", "prod": "Mobile App Checkout", "price": 6200},
    {"text": "Payment was debited but the order was cancelled automatically citing payment gateway failure.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-08-19", "prod": "Order Checkout API", "price": 1899},
    {"text": "The checkout button is unresponsive on mobile Safari when trying to submit payment.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-24", "prod": "Mobile Web Checkout", "price": 999},
    {"text": "I got an error saying transaction failed, but 10 minutes later I received an invoice email.", "sev": 3, "sent": "neutral", "rat": 3.0, "seg": "Returning User - Mobile", "date": "2026-08-28", "prod": "Order Checkout API", "price": 2400},
    {"text": "Net banking session expired during the redirect back to the merchant site, creating order confusion.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Desktop", "date": "2026-09-01", "prod": "NetBanking Gateway", "price": 3150},
    {"text": "Payment retry loop occurred: the app asked me to retry payment 3 times despite bank approvals.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-09-02", "prod": "Mobile App Checkout", "price": 4200},
    {"text": "I was charged twice because the app reported the first payment as failed when it actually succeeded.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-09-04", "prod": "Payment Processing", "price": 1750},
    {"text": "Wallet deduction happened instantly but order confirmation page threw a 504 gateway timeout.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-09-05", "prod": "Digital Wallet Service", "price": 850},
    {"text": "Payment verification takes over 2 minutes on mobile data, making me think the transaction froze.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-09-07", "prod": "Mobile Checkout Gateway", "price": 1999},
    {"text": "I received no confirmation screen after entering card CVV, just an empty white page.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-09-08", "prod": "Mobile Checkout Gateway", "price": 5600},
    {"text": "Payment failed repeatedly on mobile app today. Switched to desktop and it worked, but mobile checkout is broken.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-09-09", "prod": "Mobile App Checkout", "price": 3499},

    # Interviews (8)
    {"text": "During my first purchase on mobile, I wasn't sure if my payment went through because the screen froze on the bank redirect.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-07-18", "prod": "Mobile Checkout Gateway", "price": 2300},
    {"text": "As a first-time mobile user, the checkout flow made me anxious. The payment pending state gave zero guidance on whether I was charged.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-05", "prod": "Mobile Checkout Gateway", "price": 1500},
    {"text": "I had to retry paying three times on my smartphone. Each time I worried I would get billed multiple times.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-08-20", "prod": "Mobile Checkout Gateway", "price": 4100},
    {"text": "The payment confirmation was so delayed on the app that I accidentally submitted the order twice.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-30", "prod": "Mobile App Checkout", "price": 1800},
    {"text": "I really like the product catalog, but the checkout payment reliability on mobile is a major obstacle for new shoppers.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-09-03", "prod": "Mobile App Checkout", "price": 2750},
    {"text": "When the payment fails, the error message is too generic. It just says 'Error occurred' without saying if funds were held.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-09-06", "prod": "Payment Processing", "price": 3900},
    {"text": "I experienced checkout failure when trying to pay with Google Pay on my phone yesterday.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-09-08", "prod": "UPI Gateway Service", "price": 1150},
    {"text": "Mobile payment drop-offs are definitely happening because of the sluggish verification step.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "Power User - Mobile", "date": "2026-09-10", "prod": "Mobile Checkout Gateway", "price": 4800},

    # Surveys (7)
    {"text": "Payment needed retry twice before succeeding. The checkout process feels unstable.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-06-18", "prod": "Mobile Checkout Gateway", "price": 950},
    {"text": "The checkout failed on mobile and I could not finish my purchase.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-28", "prod": "Mobile App Checkout", "price": 2100},
    {"text": "Payment status was unclear for almost an hour after completing transaction.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-14", "prod": "Order Checkout API", "price": 1600},
    {"text": "Checkout experience was very frustrating due to multiple payment gateway errors.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-08-25", "prod": "Mobile Checkout Gateway", "price": 3300},
    {"text": "I didn't know whether my payment succeeded until I checked my bank statement.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-09-02", "prod": "Payment Processing", "price": 2800},
    {"text": "Payment failed during final step on mobile app. Please fix payment gateway.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-09-05", "prod": "Mobile App Checkout", "price": 4400},
    {"text": "Unclear order status after credit card verification on mobile browser.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-09-09", "prod": "Mobile Web Checkout", "price": 1950},
]

# 2. DELIVERY-DATE UNCERTAINTY (25 records: 14 SUP, 6 INT, 5 SUR)
# Meaningful evidence supporting "BUILD" for delivery estimation
DELIVERY_TEMPLATES = [
    # Support Tickets (14)
    {"text": "The estimated delivery date shifted by 4 days after I placed the order without any notification.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-06-15", "prod": "Delivery Tracking System", "price": 1800},
    {"text": "The tracking page said delivery by Tuesday, but customer support says it won't arrive until Friday.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-06-28", "prod": "Delivery Tracking System", "price": 3200},
    {"text": "There is a mismatch between the delivery date shown on product page and the date in order summary.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Desktop", "date": "2026-07-08", "prod": "Delivery Estimate Calculator", "price": 4500},
    {"text": "I have no idea when my package is arriving because the status has been 'Out for Delivery' for 3 days.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-07-19", "prod": "Logistics Tracker", "price": 1200},
    {"text": "Delivery date keeps changing every morning on the tracker. Need accurate arrival confirmation for travel plans.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-07-31", "prod": "Delivery Tracking System", "price": 6800},
    {"text": "The courier tracking link shows 'Invalid AWB Number' and no expected delivery date is available.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-08-04", "prod": "Logistics Tracker", "price": 950},
    {"text": "Promised 2-day delivery turned into 8 days with zero status updates on the dashboard.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-08-12", "prod": "Express Delivery Service", "price": 2900},
    {"text": "The delivery window stated 9 AM to 1 PM, but the delivery agent called at 8 PM when nobody was home.", "sev": 2, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-08-18", "prod": "Last Mile Delivery", "price": 1400},
    {"text": "Estimated arrival date disappeared completely from my active orders page.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-23", "prod": "Delivery Tracking System", "price": 2100},
    {"text": "Order was promised for birthday weekend but arrived 5 days late due to inaccurate fulfillment estimates.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-08-29", "prod": "Delivery Estimate Calculator", "price": 5100},
    {"text": "Live tracking shows driver location 20 miles away for over 5 hours with constant ETA delays.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Mobile", "date": "2026-09-02", "prod": "Logistics Tracker", "price": 3800},
    {"text": "Product page promised next day delivery, but after checkout the receipt stated 6 days delivery window.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Desktop", "date": "2026-09-04", "prod": "Delivery Estimate Calculator", "price": 4200},
    {"text": "Tracking updates are delayed by 24 hours. The package arrived before the status even changed to dispatched.", "sev": 2, "sent": "neutral", "rat": 3.0, "seg": "Returning User - Mobile", "date": "2026-09-07", "prod": "Delivery Tracking System", "price": 1600},
    {"text": "Inconsistent delivery promises across app and email confirmations cause huge planning uncertainty.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-09-09", "prod": "Delivery Estimate Calculator", "price": 7500},

    # Interviews (6)
    {"text": "The main problem for me is uncertainty around package arrival. If the app gave a reliable 2-hour window, I wouldn't worry.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-06-22", "prod": "Delivery Tracking System", "price": 2500},
    {"text": "When ordering high-value electronics, not knowing the exact delivery date makes me hesitate to purchase.", "sev": 4, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-07-14", "prod": "Delivery Estimate Calculator", "price": 12000},
    {"text": "The estimated delivery date kept bouncing between Thursday and Saturday. Clear communication would fix this.", "sev": 3, "sent": "negative", "rat": 3.0, "seg": "Returning User - Desktop", "date": "2026-08-09", "prod": "Delivery Tracking System", "price": 3100},
    {"text": "I missed my delivery because the courier arrived two days earlier than the date listed on my order dashboard.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-27", "prod": "Last Mile Delivery", "price": 1900},
    {"text": "Accurate shipping estimates are critical for regular buyers like me who need items for weekend projects.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Mobile", "date": "2026-09-03", "prod": "Delivery Estimate Calculator", "price": 6400},
    {"text": "Delivery tracking feels disconnected from actual carrier progress. Real-time milestones would build a lot of trust.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-09-08", "prod": "Logistics Tracker", "price": 2800},

    # Surveys (5)
    {"text": "Delivery date was postponed twice without explanation.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-06-29", "prod": "Delivery Tracking System", "price": 1500},
    {"text": "Arrival estimate was inaccurate by nearly a week.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-25", "prod": "Delivery Estimate Calculator", "price": 2200},
    {"text": "Delivery timeline unclear after order dispatch.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-08-15", "prod": "Logistics Tracker", "price": 3400},
    {"text": "Package arrived later than initial estimate shown during checkout.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-08-31", "prod": "Delivery Estimate Calculator", "price": 1750},
    {"text": "Estimated delivery date changed three times in 48 hours.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Mobile", "date": "2026-09-06", "prod": "Delivery Tracking System", "price": 4900},
]

# 3. ONBOARDING CONTRADICTION (20 records: 8 SUP, 7 INT, 5 SUR)
# New Users: confusion, setup friction
# Experienced / Returning Users: onboarding is straightforward and easy
ONBOARDING_TEMPLATES = [
    # Support Tickets (8)
    {"text": "As a new user, I cannot figure out how to verify my shipping address. The onboarding wizard gets stuck on step 2.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-06-10", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "The account setup instructions are confusing. I don't know where to enter my GST number or profile details.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Desktop", "date": "2026-06-24", "prod": "Account Setup Flow", "price": 0},
    {"text": "I tried signing up on mobile but never received the verification email or initial guidance on how to start.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-09", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "The introductory tutorial popups cover the search bar and cannot be dismissed on Android.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-07-21", "prod": "Mobile App Onboarding", "price": 0},
    {"text": "New registration form keeps rejecting valid passwords without clear formatting rules.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Desktop", "date": "2026-08-06", "prod": "Account Setup Flow", "price": 0},
    {"text": "I created an account for my business and setting up the team access permissions was completely seamless.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Desktop", "date": "2026-08-17", "prod": "Account Setup Flow", "price": 0},
    {"text": "The onboarding checklist made it very fast to save my default addresses and payment preferences.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Desktop", "date": "2026-08-26", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "First-time setup was very intuitive and took less than two minutes on my laptop.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Desktop", "date": "2026-09-05", "prod": "Account Setup Flow", "price": 0},

    # Interviews (7)
    {"text": "When I first joined, I was totally lost after the registration screen. It didn't tell me what step to take next.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-06-16", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "The first-time user experience felt cluttered with too many permissions requests before I could even browse items.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-07-12", "prod": "Mobile App Onboarding", "price": 0},
    {"text": "I struggled to complete onboarding on desktop because the profile completion bar wasn't interactive.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Desktop", "date": "2026-07-27", "prod": "Account Setup Flow", "price": 0},
    {"text": "As an experienced customer, the onboarding flow has always been straightforward. I never ran into any setup hurdles.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Mobile", "date": "2026-08-10", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "New users in my household had trouble signing up, but once you know where things are, the platform is easy.", "sev": 2, "sent": "neutral", "rat": 3.0, "seg": "Returning User - Mobile", "date": "2026-08-22", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "I onboarded our team last month and the whole account configuration was smooth with zero friction.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Desktop", "date": "2026-09-01", "prod": "Account Setup Flow", "price": 0},
    {"text": "Setting up my account was completely painless. Everything from phone OTP to profile creation was instant.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Mobile", "date": "2026-09-08", "prod": "Mobile App Onboarding", "price": 0},

    # Surveys (5)
    {"text": "Confusing initial step during account setup for new members.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-06-20", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "Difficult to complete onboarding on mobile browser.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-07-16", "prod": "Mobile App Onboarding", "price": 0},
    {"text": "Onboarding was fast, clear, and very well designed.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Desktop", "date": "2026-08-13", "prod": "Account Setup Flow", "price": 0},
    {"text": "Initial setup was simple and took under a minute.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Mobile", "date": "2026-08-28", "prod": "User Onboarding Wizard", "price": 0},
    {"text": "Unclear first steps after creating a new profile.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Desktop", "date": "2026-09-07", "prod": "Account Setup Flow", "price": 0},
]

# 4. SEARCH RELEVANCE (18 records: 7 SUP, 6 INT, 5 SUR)
# Stable across June-September, slightly higher among returning users
SEARCH_TEMPLATES = [
    # Support Tickets (7)
    {"text": "Searching for 'Bluetooth noise cancelling headphones' returns basic phone cases and unrelated chargers.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-06-08", "prod": "Search & Discovery Engine", "price": 2499},
    {"text": "The search filters for brand and price range reset automatically whenever I switch sorting to lowest price.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-06-26", "prod": "Catalog Search Filter", "price": 1800},
    {"text": "Search query auto-suggestions are outdated and show out-of-stock items at the top.", "sev": 2, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-07-11", "prod": "Search & Discovery Engine", "price": 3500},
    {"text": "Keyword search fails to recognize simple singular/plural spelling variations.", "sev": 2, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-07-26", "prod": "Search & Discovery Engine", "price": 1200},
    {"text": "Category filtering under 'Home Appliances' is mixing in kitchen utensils and irrelevant accessories.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-08-14", "prod": "Catalog Search Filter", "price": 4100},
    {"text": "Search results on mobile app lag by several seconds when typing queries.", "sev": 2, "sent": "negative", "rat": 3.0, "seg": "Returning User - Mobile", "date": "2026-08-27", "prod": "Search & Discovery Engine", "price": 950},
    {"text": "I searched for a specific model number and the search returned zero results even though the product is listed.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-09-06", "prod": "Search & Discovery Engine", "price": 8900},

    # Interviews (6)
    {"text": "I spend too much time filtering out irrelevant sponsored items when searching for specific kitchen appliances.", "sev": 2, "sent": "negative", "rat": 3.0, "seg": "Returning User - Desktop", "date": "2026-06-19", "prod": "Search & Discovery Engine", "price": 3200},
    {"text": "The search engine works fine for broad queries like 'shoes', but struggles with specific technical specifications.", "sev": 2, "sent": "neutral", "rat": 3.0, "seg": "Power User - Mobile", "date": "2026-07-06", "prod": "Search & Discovery Engine", "price": 5600},
    {"text": "Finding replacement parts is tricky because searching the exact part number rarely yields the matching item.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-07-30", "prod": "Search & Discovery Engine", "price": 1450},
    {"text": "I like the visual layout of search results, but the ranking algorithm pushes unrelated sponsored products to the top.", "sev": 2, "sent": "neutral", "rat": 3.0, "seg": "Returning User - Mobile", "date": "2026-08-15", "prod": "Search & Discovery Engine", "price": 2100},
    {"text": "Search relevance has been fairly consistent over the last few months, though multi-facet filtering could be improved.", "sev": 2, "sent": "neutral", "rat": 3.0, "seg": "Power User - Desktop", "date": "2026-08-31", "prod": "Catalog Search Filter", "price": 7200},
    {"text": "When I search on mobile, the sorting often resets to 'Relevance' instead of keeping my chosen 'Price: Low to High'.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-09-08", "prod": "Catalog Search Filter", "price": 1900},

    # Surveys (5)
    {"text": "Search results often contain items unrelated to my query.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-06-14", "prod": "Search & Discovery Engine", "price": 1100},
    {"text": "Filter options are reset when navigating back from a product page.", "sev": 2, "sent": "negative", "rat": 3.0, "seg": "Returning User - Desktop", "date": "2026-07-17", "prod": "Catalog Search Filter", "price": 2800},
    {"text": "Search query auto-complete is helpful but results ranking needs tuning.", "sev": 2, "sent": "neutral", "rat": 3.0, "seg": "New User - Mobile", "date": "2026-08-08", "prod": "Search & Discovery Engine", "price": 1650},
    {"text": "Search relevance is acceptable for basic terms.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Returning User - Mobile", "date": "2026-08-25", "prod": "Search & Discovery Engine", "price": 2300},
    {"text": "Difficulty finding exact product variants using search bar.", "sev": 2, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-09-04", "prod": "Search & Discovery Engine", "price": 4600},
]

# 5. WISHLIST / AI ASSISTANT FEATURE REQUESTS (15 records: 4 SUP, 6 INT, 5 SUR)
# Mostly Power Users, low severity (1-2), NO evidence of churn/revenue loss -> "DON'T BUILD YET"
WISHLIST_TEMPLATES = [
    # Support Tickets (4)
    {"text": "Feature Request: It would be nice to organize wishlist items into custom folders like 'Home' and 'Office'.", "sev": 1, "sent": "neutral", "rat": 4.0, "seg": "Power User - Desktop", "date": "2026-06-05", "prod": "Wishlist Management", "price": 0},
    {"text": "Would love an AI shopping assistant that can summarize product reviews into pros and cons.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Power User - Mobile", "date": "2026-06-27", "prod": "AI Shopping Assistant", "price": 0},
    {"text": "Suggestion: Add a side-by-side product specification comparison tool directly in the wishlist view.", "sev": 1, "sent": "neutral", "rat": 4.0, "seg": "Power User - Desktop", "date": "2026-07-15", "prod": "Product Comparison Tool", "price": 0},
    {"text": "Please consider adding price-drop alerts for items saved in my wishlist.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Power User - Mobile", "date": "2026-08-03", "prod": "Wishlist Management", "price": 0},

    # Interviews (6)
    {"text": "I think having an AI assistant that suggests matching accessories based on cart items would be a neat touch.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Power User - Mobile", "date": "2026-06-17", "prod": "AI Shopping Assistant", "price": 0},
    {"text": "An automated price history graph in the wishlist would be a cool feature for heavy shoppers like me.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Power User - Desktop", "date": "2026-07-07", "prod": "Wishlist Management", "price": 0},
    {"text": "I frequently compare three or four laptops before deciding. A native comparison feature would save me opening multiple tabs.", "sev": 2, "sent": "neutral", "rat": 3.0, "seg": "Power User - Desktop", "date": "2026-07-24", "prod": "Product Comparison Tool", "price": 0},
    {"text": "A shared family wishlist where multiple members can add items would be great for holiday shopping.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Power User - Mobile", "date": "2026-08-11", "prod": "Wishlist Management", "price": 0},
    {"text": "An AI bot that answers questions about dimensions and sizing directly from the user manual would be interesting.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Power User - Desktop", "date": "2026-08-29", "prod": "AI Shopping Assistant", "price": 0},
    {"text": "Wishlist export to CSV or PDF would be convenient for work procurement, though not urgent.", "sev": 1, "sent": "neutral", "rat": 4.0, "seg": "Power User - Desktop", "date": "2026-09-05", "prod": "Wishlist Management", "price": 0},

    # Surveys (5)
    {"text": "An AI shopping assistant to help compare specifications would be a great addition.", "sev": 1, "sent": "positive", "rat": 4.0, "seg": "Power User - Mobile", "date": "2026-06-23", "prod": "AI Shopping Assistant", "price": 0},
    {"text": "Would like ability to create multiple themed wishlists.", "sev": 1, "sent": "neutral", "rat": 4.0, "seg": "Power User - Desktop", "date": "2026-07-18", "prod": "Wishlist Management", "price": 0},
    {"text": "Side by side product comparison table would be helpful.", "sev": 1, "sent": "neutral", "rat": 4.0, "seg": "Power User - Desktop", "date": "2026-08-07", "prod": "Product Comparison Tool", "price": 0},
    {"text": "Notification when a saved wishlist product goes on sale.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Mobile", "date": "2026-08-24", "prod": "Wishlist Management", "price": 0},
    {"text": "AI recommendations for complementary products.", "sev": 1, "sent": "neutral", "rat": 4.0, "seg": "Power User - Mobile", "date": "2026-09-02", "prod": "AI Shopping Assistant", "price": 0},
]

# 6. RETURNS / REFUND FRICTION (12 records: 7 SUP, 3 INT, 2 SUR)
RETURNS_TEMPLATES = [
    # Support Tickets (7)
    {"text": "I returned my package 10 days ago but the refund status still shows 'Pending Inspection'.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-06-11", "prod": "Returns & Refund Portal", "price": 3200},
    {"text": "The courier picked up the return item last week, but the app shows pickup is still scheduled.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-06-30", "prod": "Reverse Logistics", "price": 1850},
    {"text": "Return request was rejected without any explanation despite providing photos of the damaged item.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-13", "prod": "Returns & Refund Portal", "price": 4900},
    {"text": "Refund was processed to store credit instead of my original payment method contrary to my selection.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-07-28", "prod": "Refund Processing Gateway", "price": 2100},
    {"text": "The return label barcode generated by the app was unreadable by the drop-off center scanner.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-08-16", "prod": "Returns & Refund Portal", "price": 1400},
    {"text": "Refund amount is missing the original shipping fee even though the return was due to seller defect.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-08-30", "prod": "Refund Processing Gateway", "price": 3800},
    {"text": "I have been waiting for over two weeks for customer support to approve my return authorization.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-09-06", "prod": "Returns & Refund Portal", "price": 5600},

    # Interviews (3)
    {"text": "The return initiation process is complicated. You have to navigate through four different menus just to find the return button.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-07-05", "prod": "Returns & Refund Portal", "price": 2700},
    {"text": "Refund timelines are too opaque. It says 'Refund in 5-7 business days' with no tracking milestone in between.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Mobile", "date": "2026-08-19", "prod": "Refund Processing Gateway", "price": 4200},
    {"text": "Returning an item was stressful because I had no proof of pickup until the warehouse confirmed receipt a week later.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Desktop", "date": "2026-09-04", "prod": "Reverse Logistics", "price": 3100},

    # Surveys (2)
    {"text": "Unclear refund status and delay in bank crediting.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-07-20", "prod": "Refund Processing Gateway", "price": 1900},
    {"text": "Return pickup was delayed by four days with no notification.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-08-26", "prod": "Reverse Logistics", "price": 2400},
]

# 7. PRODUCT QUALITY ISSUES (15 records: 10 SUP, 2 INT, 3 SUR)
QUALITY_TEMPLATES = [
    # Support Tickets (10)
    {"text": "The air cooler fan stopped spinning after just 3 days of use. Motor emits a burning smell.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-06-07", "prod": "Room Air Cooler 12L", "price": 4299},
    {"text": "Product arrived with deep scratches on the display screen and a cracked casing.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-06-21", "prod": "Smart Fitness Watch", "price": 2499},
    {"text": "Electric kettle heating element stopped working within two weeks. Poor durability.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Desktop", "date": "2026-07-10", "prod": "Electric Kettle 1.5L", "price": 899},
    {"text": "The Bluetooth headset left earbud has zero audio output and will not hold charge.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-23", "prod": "Wireless Bluetooth Earbuds", "price": 1599},
    {"text": "Iron plate coating started peeling off onto clothes on the very first use.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "Returning User - Mobile", "date": "2026-08-01", "prod": "Dry Iron 1000W", "price": 650},
    {"text": "Juicer mixer jar lid doesn't lock properly causing liquid to spill out during operation.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Desktop", "date": "2026-08-13", "prod": "Mixer Grinder 750W", "price": 2899},
    {"text": "Power bank capacity is significantly below advertised specs, barely charging my phone once.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "New User - Mobile", "date": "2026-08-21", "prod": "Power Bank 10000mAh", "price": 1199},
    {"text": "Soundbar produces distorted bass and buzzing noise at volumes above 50%.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Desktop", "date": "2026-08-28", "prod": "Bluetooth Soundbar 16W", "price": 2199},
    {"text": "The cricket bat willow split along the edge during the first practice session.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-09-03", "prod": "Willow Cricket Bat", "price": 1450},
    {"text": "Laptop cooling fan makes a loud grinding noise and running temperature is excessive.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Power User - Desktop", "date": "2026-09-09", "prod": "Laptop Cooling Pad", "price": 999},

    # Interviews (2)
    {"text": "I had to return a food processor last month because the motor stalled on standard vegetables. Build quality was disappointing.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "Returning User - Desktop", "date": "2026-07-16", "prod": "Food Processor 800W", "price": 4500},
    {"text": "Several small electronic items I bought recently had loose plastic parts. Quality control seems inconsistent across third-party sellers.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Power User - Mobile", "date": "2026-08-22", "prod": "Electronics Accessories", "price": 1800},

    # Surveys (3)
    {"text": "Item stopped working within one week of delivery.", "sev": 5, "sent": "negative", "rat": 1.0, "seg": "New User - Mobile", "date": "2026-07-02", "prod": "Wireless Earphones", "price": 1299},
    {"text": "Poor plastic quality and fragile build.", "sev": 3, "sent": "negative", "rat": 2.0, "seg": "Returning User - Mobile", "date": "2026-08-17", "prod": "Desk Fan", "price": 850},
    {"text": "Damaged packaging resulted in broken parts upon arrival.", "sev": 4, "sent": "negative", "rat": 1.0, "seg": "New User - Desktop", "date": "2026-09-06", "prod": "Induction Cooktop", "price": 2600},
]

# 8. POSITIVE CHECKOUT/SHOPPING EXPERIENCE (10 records: 5 SUP, 2 INT, 3 SUR)
POSITIVE_TEMPLATES = [
    # Support Tickets (5)
    {"text": "The one-click checkout was amazingly fast and my order was confirmed immediately.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Mobile", "date": "2026-06-09", "prod": "One-Click Checkout", "price": 1500},
    {"text": "Thank you for the quick resolution on my address change! Shopping experience has been great.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Desktop", "date": "2026-06-27", "prod": "Order Management Portal", "price": 2800},
    {"text": "Payment went through smoothly with Apple Pay and delivery arrived a day earlier than promised.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Mobile", "date": "2026-07-11", "prod": "Mobile Checkout Gateway", "price": 5400},
    {"text": "Very clean and user-friendly interface. Finding deals and checking out was effortless.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "New User - Desktop", "date": "2026-08-04", "prod": "Web Storefront Platform", "price": 1900},
    {"text": "Excellent packaging and lightning fast shipping. Really impressed with the service.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Desktop", "date": "2026-08-25", "prod": "Express Delivery Service", "price": 3700},

    # Interviews (2)
    {"text": "Whenever I use the desktop website, the checkout process is seamless. Saved cards and addresses work reliably every time.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Desktop", "date": "2026-07-03", "prod": "Web Checkout Gateway", "price": 4200},
    {"text": "I buy household essentials every month and the reordering flow is super convenient.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Mobile", "date": "2026-08-18", "prod": "Mobile App Storefront", "price": 2600},

    # Surveys (3)
    {"text": "Checkout was smooth and fast with no issues.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Returning User - Mobile", "date": "2026-06-18", "prod": "Mobile Checkout Gateway", "price": 1200},
    {"text": "Great shopping experience and prompt delivery.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "New User - Mobile", "date": "2026-07-22", "prod": "Web Storefront Platform", "price": 2100},
    {"text": "Payment and order confirmation worked perfectly.", "sev": 1, "sent": "positive", "rat": 5.0, "seg": "Power User - Mobile", "date": "2026-08-30", "prod": "Mobile Checkout Gateway", "price": 3500},
]


def assemble_all_records() -> List[Dict]:
    """Assembles all 150 synthetic records from scenario templates and categorizes by source."""
    theme_buckets = [
        ("Payment & Checkout Reliability", "checkout", PAYMENT_TEMPLATES, 20, 8, 7),
        ("Delivery-Date Uncertainty", "delivery", DELIVERY_TEMPLATES, 14, 6, 5),
        ("Onboarding Confusion", "onboarding", ONBOARDING_TEMPLATES, 8, 7, 5),
        ("Search Relevance", "search", SEARCH_TEMPLATES, 7, 6, 5),
        ("Wishlist / AI Assistant Feature Requests", "wishlist", WISHLIST_TEMPLATES, 4, 6, 5),
        ("Returns / Refund Friction", "returns", RETURNS_TEMPLATES, 7, 3, 2),
        ("Product Quality Issues", "product_quality", QUALITY_TEMPLATES, 10, 2, 3),
        ("Positive Checkout/Shopping Experience", "checkout", POSITIVE_TEMPLATES, 5, 2, 3),
    ]

    support_records: List[Dict] = []
    interview_records: List[Dict] = []
    survey_records: List[Dict] = []

    for theme_name, product_area, templates, n_sup, n_int, n_sur in theme_buckets:
        assert len(templates) == (n_sup + n_int + n_sur), (
            f"Template count mismatch for theme '{theme_name}': expected {n_sup+n_int+n_sur}, got {len(templates)}"
        )

        sup_slice = templates[:n_sup]
        int_slice = templates[n_sup : n_sup + n_int]
        sur_slice = templates[n_sup + n_int :]

        for item in sup_slice:
            record = dict(item)
            record["theme"] = theme_name
            record["product_area"] = product_area
            record["source_type"] = "Support Ticket"
            record["source_name"] = "Synthetic Support Dataset"
            support_records.append(record)

        for item in int_slice:
            record = dict(item)
            record["theme"] = theme_name
            record["product_area"] = product_area
            record["source_type"] = "Interview"
            record["source_name"] = "Synthetic Customer Interviews"
            interview_records.append(record)

        for item in sur_slice:
            record = dict(item)
            record["theme"] = theme_name
            record["product_area"] = product_area
            record["source_type"] = "Survey"
            record["source_name"] = "Synthetic Customer Survey"
            survey_records.append(record)

    # Verify exact required partition
    assert len(support_records) == 75, f"Expected 75 support records, got {len(support_records)}"
    assert len(interview_records) == 40, f"Expected 40 interview records, got {len(interview_records)}"
    assert len(survey_records) == 35, f"Expected 35 survey records, got {len(survey_records)}"

    # Assign IDs
    final_list: List[Dict] = []
    for i, r in enumerate(support_records, start=1):
        r["feedback_id"] = f"SYN-SUP-{i:03d}"
        final_list.append(r)

    for i, r in enumerate(interview_records, start=1):
        r["feedback_id"] = f"SYN-INT-{i:03d}"
        final_list.append(r)

    for i, r in enumerate(survey_records, start=1):
        r["feedback_id"] = f"SYN-SUR-{i:03d}"
        final_list.append(r)

    return final_list


def generate_synthetic_feedback(
    output_path: Union[str, Path] = "data/processed/voc_synthetic_feedback.csv",
    random_state: int = 42
) -> pd.DataFrame:
    """Generates the canonical 150-record synthetic evidence dataset and saves it to output_path."""
    out_p = Path(output_path)
    records = assemble_all_records()

    rows = []
    for r in records:
        rows.append({
            "feedback_id": r["feedback_id"],
            "source_type": r["source_type"],
            "source_name": r["source_name"],
            "feedback_text": r["text"],
            "created_at": r["date"],
            "customer_segment": r["seg"],
            "product_area": r["product_area"],
            "sentiment": r["sent"],
            "severity": int(r["sev"]),
            "language": "en",
            "data_type": "synthetic",
            "metadata_origin": METADATA_ORIGIN_SYNTHETIC,
            "rating": float(r["rat"]),
            "product_name": r["prod"],
            "product_price": int(r["price"]),
            "theme": r["theme"]  # temporary for report
        })

    df = pd.DataFrame(rows)

    # Quality Assertions
    assert len(df) == 150, f"Expected 150 rows, got {len(df)}"
    assert df["feedback_id"].is_unique, "feedback_id is not unique"
    assert (df["data_type"] == "synthetic").all(), "All records must have data_type == 'synthetic'"
    assert (df["language"] == "en").all(), "All records must have language == 'en'"
    assert df["feedback_text"].notnull().all() and (df["feedback_text"].str.strip() != "").all()
    assert df["created_at"].notnull().all()
    assert df["customer_segment"].notnull().all()
    assert df["product_area"].notnull().all()
    assert df["sentiment"].isin(["positive", "negative", "neutral"]).all()
    assert df["severity"].isin([1, 2, 3, 4, 5]).all()

    # Source breakdown assertion
    src_counts = df["source_type"].value_counts()
    assert src_counts.get("Support Ticket", 0) == 75, "Expected 75 Support Tickets"
    assert src_counts.get("Interview", 0) == 40, "Expected 40 Interviews"
    assert src_counts.get("Survey", 0) == 35, "Expected 35 Surveys"

    # Date range assertion
    min_date = df["created_at"].min()
    max_date = df["created_at"].max()
    assert min_date >= "2026-06-01", f"Date {min_date} is before 2026-06-01"
    assert max_date <= "2026-09-10", f"Date {max_date} is after 2026-09-10"

    # Add month column for summary
    df["month"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m")

    # Final dataframe format
    theme_distribution = df["theme"].value_counts()
    df_output = df[TARGET_COLUMNS].copy()

    # Save to disk
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(out_p, index=False)

    # Print Report
    print("=" * 70)
    print("SYNTHETIC FEEDBACK GENERATION REPORT")
    print("=" * 70)
    print(f"Total synthetic records:             {len(df_output):,}")
    print(f"Data type flag:                      {df_output['data_type'].iloc[0]}")
    print(f"Date range:                          {min_date} to {max_date}")
    
    print("\nSource Type Distribution:")
    for stype, cnt in df_output["source_type"].value_counts().items():
        print(f"  - {stype:20s}: {cnt:3d} ({cnt/len(df_output)*100:.1f}%)")

    print("\nMajor Themes Distribution:")
    for th, cnt in theme_distribution.items():
        print(f"  - {th:42s}: {cnt:3d} ({cnt/len(df_output)*100:.1f}%)")

    print("\nProduct Area Distribution:")
    for pa, cnt in df_output["product_area"].value_counts().items():
        print(f"  - {pa:20s}: {cnt:3d} ({cnt/len(df_output)*100:.1f}%)")

    print("\nCustomer Segment Distribution:")
    for seg, cnt in df_output["customer_segment"].value_counts().items():
        print(f"  - {seg:26s}: {cnt:3d} ({cnt/len(df_output)*100:.1f}%)")

    print("\nSentiment Distribution:")
    for sent, cnt in df_output["sentiment"].value_counts().items():
        print(f"  - {sent:20s}: {cnt:3d} ({cnt/len(df_output)*100:.1f}%)")

    print("\nSeverity Distribution (1-5):")
    for sev, cnt in df_output["severity"].value_counts().sort_index().items():
        print(f"  - Level {sev}:                {cnt:3d} ({cnt/len(df_output)*100:.1f}%)")

    print("\nTemporal Distribution by Month:")
    for mth, cnt in df["month"].value_counts().sort_index().items():
        print(f"  - {mth}:                    {cnt:3d} ({cnt/len(df_output)*100:.1f}%)")

    print("\nPayment Theme Monthly Trend (Increasing pattern):")
    pay_m = df[df["theme"] == "Payment & Checkout Reliability"]["month"].value_counts().sort_index()
    for mth, cnt in pay_m.items():
        print(f"  - {mth}:                    {cnt:3d} records")

    print("=" * 70)
    print(f"Output saved to: {out_p}")

    return df_output


if __name__ == "__main__":
    generate_synthetic_feedback()
