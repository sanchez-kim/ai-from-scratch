"""Ten short passages written for this experiment (MIT, like the rest of the repo).

They mix registers on purpose: a story, a recipe, code, a support reply, a
definition, a travel note, a list, a dialogue, a news-style line and a poem.
The model reads each one token by token, and at every position I record what
it expected next.
"""

PASSAGES = [
    "The lighthouse keeper counted the ships every evening. On the night of the "
    "storm he counted seven, but only six reached the harbour. He walked down to "
    "the rocks with a lamp and waited until the rain stopped, listening for a "
    "voice in the water.",

    "To make the dough, mix 500 grams of flour with 10 grams of salt and 7 grams "
    "of dry yeast. Add 320 millilitres of warm water and knead for ten minutes, "
    "until the dough is smooth. Cover the bowl and leave it in a warm place for "
    "an hour.",

    "def average(values):\n"
    "    \"\"\"Return the mean of a list of numbers.\"\"\"\n"
    "    if not values:\n"
    "        raise ValueError(\"values must not be empty\")\n"
    "    total = 0\n"
    "    for value in values:\n"
    "        total += value\n"
    "    return total / len(values)\n",

    "Thank you for contacting us about your order. I am sorry that the package "
    "arrived late. I have checked the tracking number, and the delay happened at "
    "the regional sorting centre. As a refund for the shipping cost, we have added "
    "a credit to your account.",

    "A prime number is a whole number greater than one that has no divisors other "
    "than one and itself. The first few primes are 2, 3, 5, 7 and 11. Every whole "
    "number greater than one can be written as a product of primes in exactly one "
    "way.",

    "We left the village before sunrise and followed the river north. By noon the "
    "path had turned into a narrow trail between pine trees, and we could hear the "
    "waterfall long before we saw it. My sister wanted to swim, but the water was "
    "far too cold.",

    "Things to pack for a weekend in the mountains:\n"
    "- a warm jacket and a hat\n"
    "- two pairs of wool socks\n"
    "- a water bottle\n"
    "- a map of the trails\n"
    "- sunscreen, even if it is cloudy\n"
    "- a small first aid kit\n",

    "\"Did you lock the back door?\" Maria asked.\n"
    "\"I think so,\" said Tom. \"Why?\"\n"
    "\"Because the cat is outside again, and it was inside when we went to bed.\"\n"
    "Tom put down his book and looked at the window.",

    "The city council voted on Tuesday to extend the night bus service until two "
    "in the morning. The change starts next month and will cover four routes. "
    "Council members said the decision followed a survey in which most residents "
    "asked for later buses.",

    "The kettle hums before it sings,\n"
    "the window holds the morning light,\n"
    "and in the quiet of small things\n"
    "the house forgets the long cold night.\n"
    "A spoon, a cup, a slice of bread,\n"
    "the day begins with what was said.\n",
]
