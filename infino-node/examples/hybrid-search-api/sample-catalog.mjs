// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// A small, self-contained product catalog bundled with the example. Interactive
// runs pull a live sample from the HuggingFace Hub; this is the offline fallback
// used when the Hub is unreachable, and the fixed catalog the SMOKE self-check
// indexes so the end-to-end gate stays deterministic and never depends on a
// third-party service being up.
//
// Each entry is already in the shape the example indexes: `text` (title +
// description) is what gets tokenized for BM25 and embedded for vector search.

export const SAMPLE_CATALOG = [
  { title: "Stainless Steel Chef's Knife", text: "Stainless Steel Chef's Knife. An 8-inch high-carbon blade with an ergonomic handle for precise slicing, dicing, and mincing in the kitchen.", price: 39.99, rating: 4.7, category: "Kitchen" },
  { title: "Pre-Seasoned Cast Iron Skillet", text: "Pre-Seasoned Cast Iron Skillet. A 12-inch pan that goes from stovetop to oven, ideal for searing, frying, and baking with even heat retention.", price: 29.95, rating: 4.8, category: "Kitchen" },
  { title: "Silicone Cooking Utensil Set", text: "Silicone Cooking Utensil Set. A heat-resistant set of spatulas, spoons, and a ladle that won't scratch nonstick cookware.", price: 24.5, rating: 4.5, category: "Kitchen" },
  { title: "Digital Kitchen Scale", text: "Digital Kitchen Scale. A compact scale with gram precision for baking and portioning, with a tare button and easy-clean surface.", price: 15.99, rating: 4.6, category: "Kitchen" },
  { title: "French Press Coffee Maker", text: "French Press Coffee Maker. A 34-ounce borosilicate glass press for rich, full-bodied coffee brewed in four minutes.", price: 27.0, rating: 4.4, category: "Kitchen" },
  { title: "Bamboo Cutting Board Set", text: "Bamboo Cutting Board Set. Three durable, knife-friendly boards in graduated sizes with juice grooves for meal prep.", price: 22.99, rating: 4.6, category: "Kitchen" },
  { title: "The Art of Simple Cooking", text: "The Art of Simple Cooking. A beginner-friendly cookbook with 120 weeknight recipes and techniques for anyone who loves to cook.", price: 18.75, rating: 4.7, category: "Books" },
  { title: "Hydrating Facial Moisturizer", text: "Hydrating Facial Moisturizer. A lightweight daily cream with hyaluronic acid that keeps skin soft and moisturized without a greasy feel.", price: 16.49, rating: 4.5, category: "Beauty" },
  { title: "Shea Butter Body Lotion", text: "Shea Butter Body Lotion. A rich, fast-absorbing lotion that deeply nourishes dry skin and locks in moisture all day.", price: 12.99, rating: 4.6, category: "Beauty" },
  { title: "Vitamin C Brightening Serum", text: "Vitamin C Brightening Serum. An antioxidant serum that evens skin tone and adds a healthy glow while hydrating.", price: 21.0, rating: 4.3, category: "Beauty" },
  { title: "Aloe Vera Soothing Gel", text: "Aloe Vera Soothing Gel. A cooling, moisturizing gel for sunburn relief and everyday hydration on face and body.", price: 9.99, rating: 4.5, category: "Beauty" },
  { title: "Nourishing Lip Balm Variety Pack", text: "Nourishing Lip Balm Variety Pack. A set of six beeswax balms that keep lips soft and moisturized through dry weather.", price: 8.49, rating: 4.4, category: "Beauty" },
  { title: "Scented Soy Candle Gift Set", text: "Scented Soy Candle Gift Set. A boxed trio of hand-poured candles — vanilla, lavender, and sea salt — a thoughtful birthday gift.", price: 26.99, rating: 4.8, category: "Home" },
  { title: "Cozy Weighted Blanket", text: "Cozy Weighted Blanket. A 15-pound plush blanket that eases stress and improves sleep, a comforting present for any occasion.", price: 49.99, rating: 4.7, category: "Home" },
  { title: "Essential Oil Diffuser", text: "Essential Oil Diffuser. An ultrasonic aromatherapy diffuser with color-changing light for a calming home atmosphere.", price: 23.5, rating: 4.4, category: "Home" },
  { title: "Memory Foam Pillow", text: "Memory Foam Pillow. A contoured pillow that supports the neck and shoulders for a restful night's sleep.", price: 34.0, rating: 4.5, category: "Home" },
  { title: "Wireless Bluetooth Earbuds", text: "Wireless Bluetooth Earbuds. Compact earbuds with rich sound, a charging case, and all-day battery — a popular gift for music lovers.", price: 45.99, rating: 4.3, category: "Electronics" },
  { title: "Portable Power Bank", text: "Portable Power Bank. A 20000mAh charger that refuels a phone several times over, perfect for travel and commuting.", price: 28.99, rating: 4.6, category: "Electronics" },
  { title: "Smart LED Light Bulbs", text: "Smart LED Light Bulbs. A four-pack of app-controlled bulbs with millions of colors and voice-assistant support.", price: 32.0, rating: 4.4, category: "Electronics" },
  { title: "Noise-Cancelling Headphones", text: "Noise-Cancelling Headphones. Over-ear wireless headphones with active noise cancellation and 30-hour battery, a premium birthday gift.", price: 89.99, rating: 4.7, category: "Electronics" },
  { title: "Mindfulness Guided Journal", text: "Mindfulness Guided Journal. A daily journal with prompts for gratitude and reflection to build a calmer routine.", price: 14.25, rating: 4.6, category: "Books" },
  { title: "Wooden Building Blocks Set", text: "Wooden Building Blocks Set. A 100-piece set of smooth hardwood blocks that sparks creative play — a classic gift for kids.", price: 33.5, rating: 4.8, category: "Toys" },
  { title: "Non-Slip Yoga Mat", text: "Non-Slip Yoga Mat. A cushioned, eco-friendly mat with a carrying strap for yoga, pilates, and home workouts.", price: 25.99, rating: 4.5, category: "Sports" },
  { title: "Insulated Stainless Water Bottle", text: "Insulated Stainless Water Bottle. A 24-ounce bottle that keeps drinks cold for 24 hours and hot for 12, leak-proof for the gym.", price: 19.99, rating: 4.7, category: "Sports" },
  { title: "Fountain Pen Gift Box", text: "Fountain Pen Gift Box. An elegant boxed fountain pen with converter and ink cartridges, a refined gift for writers.", price: 37.0, rating: 4.5, category: "Office" },
  { title: "Leather Journal Notebook", text: "Leather Journal Notebook. A refillable hand-bound journal with thick cream paper for writing, sketching, and planning.", price: 21.5, rating: 4.6, category: "Office" },
  { title: "Gourmet Chocolate Assortment", text: "Gourmet Chocolate Assortment. A boxed collection of dark and milk chocolate truffles — a crowd-pleasing birthday gift.", price: 24.0, rating: 4.7, category: "Grocery" },
  { title: "Herbal Tea Sampler", text: "Herbal Tea Sampler. A caffeine-free assortment of chamomile, peppermint, and berry teas in a giftable tin.", price: 17.49, rating: 4.4, category: "Grocery" },
  { title: "Indoor Herb Garden Kit", text: "Indoor Herb Garden Kit. A self-watering planter with basil, cilantro, and parsley seeds — fresh herbs for anyone who loves cooking.", price: 30.99, rating: 4.3, category: "Garden" },
  { title: "Ceramic Nonstick Frying Pan", text: "Ceramic Nonstick Frying Pan. A 10-inch pan with a toxin-free ceramic coating for easy, low-oil cooking and quick cleanup.", price: 26.5, rating: 4.4, category: "Kitchen" },
];
