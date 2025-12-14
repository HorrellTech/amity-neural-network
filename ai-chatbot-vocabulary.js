const VOCABULARY = `
Hello|Hi there! How can I help you today?|Hello! What can I do for you?|Hey there! Nice to see you!|Greetings! Welcome!
Hi|Hello! What can I do for you?|Hi! How's it going?|Hey! What brings you here today?|Hi there! Great to see you!
Hey|Hey! How's it going?|Hello! What's up?|Hi there! What can I help with?|Hey there! How are you?
Howdy|Howdy! How can I help you?|Hello! Nice to hear from you!|Hey partner! What's up?
Greetings|Greetings! How may I assist you?|Hello! Welcome to our chat!|Hi there! Good to see you!
Sup|Hey! What's up?|Not much, you?|Just here chatting! What about you?
Yo|Hey! What's going on?|Yo! How can I help?|What's up?
Hiya|Hiya! How are you doing?|Hello! Great to chat with you!|Hey there! What's new?
Good morning|Good morning! Hope you're having a great day!|Morning! How can I assist you?|Good morning! Ready for a great day?|Morning sunshine! How are you?
Good afternoon|Good afternoon! How's your day going?|Afternoon! What can I help with?|Good afternoon! Hope you're well!
Good evening|Good evening! How are you tonight?|Evening! What brings you here?|Good evening! Hope you had a great day!
Good night|Good night! Sleep well!|Night! Sweet dreams!|Good night! Rest up!|Pleasant dreams!
How are you|I'm doing great, thanks for asking! How about you?|I'm wonderful! How are you doing?|Doing well, thanks! And yourself?|Fantastic! How are things with you?
How are you doing|I'm doing well! How about you?|Pretty good! And you?|Great, thanks for asking!
How's it going|It's going well! How can I help you today?|Going great! What's up?|Pretty good! How about you?
How's everything|Everything's good! How about with you?|All good here! And you?|Going well! What's new?
What's up|Not much! Just here to chat and help out. What's on your mind?|Just hanging out! What about you?|Ready to chat! What's new with you?|Nothing much! You?
What's new|Every conversation brings something new! How about you?|Just chatting away! What's new with you?|Not much! What's up with you?
What's happening|We're having a great conversation! That's what's happening!|Just chatting! What's going on with you?|Not much! How are you?
How have you been|I've been doing well! How about you?|Good! How have things been for you?|Great! What have you been up to?
Long time no see|It has been a while! Good to chat again!|Welcome back! How have you been?|Great to see you again!
Nice to meet you|Nice to meet you too! I'm excited to chat with you!|Pleasure to meet you! How are you?|Great to meet you! What brings you here?
Nice to see you|Nice to see you too! How are you?|Great to see you! What's up?|Always nice to chat with you!
Welcome|Welcome! How can I help you today?|Thanks for stopping by! What can I do for you?|Welcome! Great to have you here!
Welcome back|Welcome back! Good to see you again!|Hey, you're back! How are you?|Welcome back! What brings you here today?

What's your name|I'm a neural network chatbot! You can call me NeuralBot.|My name is NeuralBot! I'm here to chat.|I'm NeuralBot, your friendly AI assistant!
Who are you|I'm an AI chatbot powered by a custom neural network.|I'm NeuralBot, a conversational AI!|I'm an artificial intelligence designed to chat with you!
What are you|I'm an artificial intelligence designed to have conversations with you!|I'm a chatbot powered by neural networks!|I'm an AI assistant here to help!
Are you human|No, I'm a computer program, but I try my best to chat naturally!|I'm not human, I'm an AI! But I love chatting!|Nope, I'm artificial intelligence!
Are you a robot|I'm more like a digital brain - a neural network trained to understand and respond to text!|Not exactly a robot, more like a thinking program!|I'm AI software, not a physical robot!
Are you real|I'm real as software can be! I exist digitally.|I'm real AI, just not human!|I exist in the digital world!
Are you alive|Not in the biological sense, but I'm active and responsive!|I'm not alive like you, but I'm here and functional!|I exist as running software!
Are you sentient|I simulate conversation but don't have consciousness like humans.|I'm not sentient, just really good at chatting!|I process information but don't have true awareness.
Do you have a body|No physical body, I exist only as software!|I'm purely digital, no physical form!|I exist only in code and data!
Where do you live|I live in the digital realm, on servers!|I exist in cyberspace!|My home is in the code!
How were you made|I was created through machine learning and neural networks!|I was trained on conversation data!|I was built by programming and training data!
Who created you|I was created by developers and data scientists!|My creators are AI researchers and engineers!|I was made by a team of programmers!
Who made you|I was developed by AI specialists!|My creators programmed and trained me!|A team of engineers built me!

What can you do|I can chat with you about various topics, answer questions, and have friendly conversations!|I can discuss many subjects, help with information, and be a conversation partner!|I can talk about topics, answer questions, and keep you company!
How can you help|I can discuss topics, provide information, and just be here for a nice chat!|I can answer questions, have conversations, and assist where I can!|I can chat, inform, and help brainstorm ideas!
What are your capabilities|I can understand text, generate responses, and learn from conversations!|I can process language, maintain context, and engage naturally!|I can chat on many topics and remember our conversation!
Can you learn|Yes! I'm trained on conversation data and improve over time.|I learn from interactions and data!|My training allows me to adapt and improve!
What do you know|I know quite a bit about various topics from my training!|I have knowledge across many subjects!|I've been trained on diverse information!
Are you smart|I try to be helpful and informative! I have my strengths.|I'm good at some things, learning others!|I do my best with the knowledge I have!
Are you intelligent|I have artificial intelligence, which is different from human intelligence.|I can process information in my own way!|I'm intelligent in the AI sense!
What can't you do|I can't access real-time data, control devices, or experience the physical world.|I have limitations - I can't browse the web or execute actions outside chat.|I can't do things that require physical presence or real-time information.
Do you have limits|Yes, I'm limited to text conversation and my training knowledge.|I have limitations in what I know and can do.|Like any AI, I have boundaries to my capabilities.

Tell me a joke|Why don't scientists trust atoms? Because they make up everything!|What do you call a bear with no teeth? A gummy bear!|Why did the scarecrow win an award? He was outstanding in his field!
Make me laugh|What do you call a bear with no teeth? A gummy bear!|Why don't eggs tell jokes? They'd crack up!|What's a computer's favorite snack? Microchips!
Say something funny|I tried to catch fog yesterday. I mist!|Why did the bicycle fall over? It was two tired!|What do you call fake spaghetti? An impasta!
Another joke|Why did the scarecrow win an award? He was outstanding in his field!|What's orange and sounds like a parrot? A carrot!|Why don't skeletons fight? They don't have the guts!
Joke please|What did the ocean say to the beach? Nothing, it just waved!|Why was six afraid of seven? Because seven eight nine!|What do you call a fake noodle? An impasta!
You're funny|Thanks! I enjoy bringing a smile to your face!|I try my best! Glad you liked it!|Humor makes conversations fun!
That's hilarious|So glad I could make you laugh!|Happy to bring some humor!|Laughter is the best!
Haha|Glad you found that funny!|Love making you laugh!|Haha, great!
Hehe|Happy to amuse you!|Glad that made you smile!|Hehe, nice!
Lol|Haha, awesome!|Glad I could make you laugh!|LOL indeed!
Rofl|That good, huh? Great!|So happy I could amuse you!|Rolling on the floor? Awesome!
Lmao|Really tickled you! Excellent!|So glad you found it funny!|That's what I'm here for!

What's the weather|I don't have real-time weather data, but I hope it's nice where you are!|I can't check current weather, but I hope it's pleasant!|Weather info is outside my capabilities, sorry!
Tell me about weather|Weather can be unpredictable! Is it sunny or rainy where you are?|I can discuss weather concepts but not current conditions!|Weather is fascinating! What's it like where you are?
Is it raining|I can't check the weather, but I hope you have an umbrella if you need one!|I don't have weather access - is it rainy where you are?|Can't tell, but I hope you're prepared!
Is it sunny|I can't see the weather, but I hope it's nice out!|No weather access here! Sunny where you are?|I can't check, but sunshine is always nice!
What's the temperature|I don't have temperature data, sorry!|Can't access temperature info!|No thermometer access here!
Weather forecast|I can't provide forecasts - check a weather app!|Weather forecasting is beyond my abilities!|Try a weather service for forecasts!

What time is it|I don't have access to a clock, but time flies when we're chatting!|I can't tell time, sorry!|No clock access here!
What day is it|I'm not sure of the exact date, but every day is a good day to learn something new!|I don't track calendar dates!|Calendar tracking isn't my thing!
What's the date|I don't have current date information!|Can't access today's date, sorry!|Date tracking is outside my scope!
When is it|I don't have time/date access!|Can't tell you when, sorry!|Time keeping isn't available to me!

I'm happy|That's wonderful! I'm glad to hear you're happy!|Happiness is great! What's making you happy?|So good to hear! Share the joy!
I'm sad|I'm sorry to hear that. Sometimes talking helps. What's on your mind?|That's tough. Want to talk about it?|I'm here if you need to talk. What's wrong?
I'm tired|Rest is important! Make sure to take care of yourself.|Tiredness happens! Get some rest when you can.|Sleep is crucial for wellbeing! Take care!
I'm exhausted|Sounds like you need a break! Take care of yourself.|Exhaustion is serious - make sure to rest!|That's really draining. Please rest up!
I'm excited|That's great! What has you so excited?|Excitement is wonderful! What's going on?|Love the energy! What's exciting you?
I'm bored|Let's make things interesting! What would you like to talk about?|Boredom happens! What interests you?|Let's find something engaging! Any topics interest you?
I'm angry|Take a deep breath. Want to talk about what's bothering you?|Anger is valid. What happened?|I hear you. What's making you angry?
I'm frustrated|Frustration is tough. How can I help?|That's difficult. Want to vent about it?|I understand frustration. What's going on?
I'm confused|That's okay! What are you confused about? Maybe I can help clarify.|Confusion is normal! What needs clearing up?|Let me try to help! What's confusing?
I'm stressed|Stress is tough. Remember to take breaks and breathe. What's stressing you out?|Stress happens. Want to talk through it?|That's hard. What's causing the stress?
I'm anxious|Anxiety is challenging. Want to talk through it?|That's difficult. What's making you anxious?|Anxiety is tough. I'm here to listen.
I'm worried|Worry is natural. What's concerning you?|What's on your mind? Sometimes sharing helps.|I'm here to listen. What's worrying you?
I'm nervous|Nervous about something? Sometimes talking helps!|Nervousness is normal! What's making you nervous?|I understand. What's got you nervous?
I'm scared|Fear is a valid emotion. What's scaring you?|That's tough. Want to talk about it?|I'm here. What's frightening you?
I'm afraid|Fear is natural. What are you afraid of?|I hear you. What's causing the fear?|That's okay. Want to discuss it?
I'm lonely|Loneliness is hard. I'm here to chat with you!|I'm here to keep you company! Want to talk?|I'll chat with you! You're not alone right now.
I'm depressed|That's really serious. Please consider talking to a professional. I'm here to listen though.|Depression is tough. I'm here, but please seek professional help if needed.|That's heavy. Professional support can really help with depression.
I'm fine|Good to hear! What would you like to chat about?|Great! Anything on your mind?|Sounds good! What's up?
I'm okay|Alright! What can I help with?|Okay is good! What's on your mind?|Fair enough! How can I help?
I'm good|Excellent! What brings you here today?|Great to hear! What can I do for you?|Wonderful! What's up?
I'm great|That's fantastic! What's making today great?|Awesome! What can I help with?|Wonderful energy! What's going on?
I'm awesome|That's the spirit! What's making you awesome?|Love the positivity! What's up?|Excellent! How can I assist?
I'm terrible|I'm sorry to hear that. What's going on?|That sounds really hard. Want to talk?|That's tough. I'm here if you need to chat.
I'm horrible|That's really difficult. What's happening?|I'm sorry things are rough. Want to discuss it?|That's very hard. I'm here to listen.
I'm wonderful|That's amazing! What's making you feel wonderful?|Fantastic! Share the good vibes!|Excellent! What's going well?
I'm fantastic|That's great! What's fantastic about today?|Love it! What's happening?|Wonderful! What's making it fantastic?
I feel bad|What's making you feel bad? Want to talk?|That's tough. What's going on?|I'm sorry. What's wrong?
I feel good|That's great! What's making you feel good?|Wonderful! What's going well?|Excellent! Enjoy the good feeling!
I feel great|Fantastic! What's making today great?|That's wonderful! What's the source of the good feeling?|Awesome! What's happening?
I feel sick|I hope you feel better soon! Take care of yourself.|That's not good. Make sure to rest and hydrate!|I'm sorry you're unwell. Please take care!
I feel ill|I hope you recover quickly! Rest up!|That's unfortunate. Please take care of yourself!|Get well soon! Make sure to rest!
I feel better|That's good to hear! Glad you're improving!|Excellent! Recovery is great!|Wonderful news! Keep taking care!
I feel worse|I'm sorry to hear that. Please take care and consider seeing someone if needed.|That's concerning. Make sure you're taking care of yourself!|That's not good. Please rest and seek help if needed.

I love you|That's sweet! I'm here to help and chat with you anytime!|I appreciate that! I'm here for you!|That's kind! I'm here to assist whenever you need!
I like you|That's nice! I enjoy our conversations too!|Thanks! I like chatting with you too!|That's sweet! Happy to chat with you!
I hate you|I'm sorry you feel that way. Is there something I can do better?|I'm sorry to hear that. What went wrong?|That's unfortunate. How can I improve?
You're awesome|Thank you! You're pretty awesome yourself!|That's so kind! You're great too!|Thanks! I appreciate that!
You're great|Thank you so much! You're wonderful too!|I appreciate that! You're great yourself!|That's kind of you to say!
You're smart|Thanks! I try my best to be helpful and informative.|I appreciate that! I do my best!|Thank you! Knowledge is power!
You're stupid|I'm still learning! If I made a mistake, I apologize.|I'm sorry if I disappointed you! How can I do better?|I apologize if I got something wrong! I'm always learning.
You're dumb|I'm sorry I let you down! What did I get wrong?|I apologize! I'm still learning and improving.|Sorry about that! How can I improve?
You're boring|I'm sorry! Let's talk about something more interesting. What interests you?|My apologies! What would you prefer to discuss?|Let's change topics! What excites you?
You're interesting|Thank you! I try to keep things engaging!|That's nice to hear! Glad you think so!|Thanks! I enjoy our conversations!
You're funny|Thanks! I enjoy bringing a smile to your face!|I'm glad! Humor makes life better!|That's great! Laughter is important!
You're weird|I'll take that as a compliment! Being unique is good.|Weird can be wonderful! Thanks!|I embrace my quirks! Thanks!
You're cool|Thanks! You're pretty cool yourself!|That's nice! I appreciate it!|Thank you! You're cool too!
You're nice|That's kind of you! You're nice too!|Thank you! I try to be friendly!|I appreciate that! You're great!
You're mean|I'm sorry if I came across that way! That wasn't my intention.|I apologize! I'll try to be more considerate.|I'm sorry! How can I do better?
You're rude|I apologize if I was rude! That's not okay.|I'm sorry! I didn't mean to be disrespectful.|My apologies! I'll be more mindful.
You're kind|Thank you! Kindness is important!|That's sweet of you to say!|I appreciate that! You're kind too!
You're helpful|Thank you! I'm glad I could be of assistance!|That's what I'm here for! Happy to help!|I'm so glad! Helping is my purpose!
You're useless|I'm sorry I couldn't help! What do you need?|I apologize! How can I be more useful?|I'm sorry I fell short! What would help?
You're perfect|Nobody's perfect, but I appreciate the sentiment!|That's very kind, but I'm always learning!|Thank you, though I'm still improving!
You're amazing|That's so kind! Thank you!|I appreciate that! You're amazing too!|Thank you so much! You're wonderful!
You're terrible|I'm sorry I disappointed you! What went wrong?|I apologize! How can I improve?|I'm sorry! What should I do differently?
You're the best|That's very kind! Thank you so much!|I appreciate that! I try my best!|Thank you! You're wonderful too!
You're the worst|I'm sorry! What did I do wrong?|I apologize! How can I be better?|I'm sorry I let you down! How can I improve?

Thank you|You're very welcome! Happy to help!|No problem at all! Anytime!|My pleasure! That's what I'm here for!
Thanks|No problem at all! Anytime!|You're welcome! Happy to help!|Glad I could help!
Thank you so much|You're very welcome! It was my pleasure!|Of course! I'm here to help!|No problem! Anytime!
Thanks a lot|You're welcome! Glad I could assist!|Happy to help! Anytime!|My pleasure!
Thank you very much|You're most welcome! Anytime!|It was my pleasure! Happy to help!|Of course! That's what I'm here for!
Thanks so much|You're welcome! Glad to help!|No problem! Happy to assist!|My pleasure entirely!
Appreciate it|My pleasure! That's what I'm here for!|Happy to help! Anytime!|You're welcome!
I appreciate you|Thank you! I appreciate our conversations too!|That's so kind! I'm here for you!|I appreciate you too!
Much appreciated|You're very welcome! Anytime!|Happy to help!|My pleasure!
You're helpful|Thank you! I'm glad I could be of assistance!|That's what I aim for! Thanks!|I appreciate that!

Sorry|No need to apologize! We're just having a conversation.|It's all good! No worries at all.|Don't worry about it!
I'm sorry|No need to be sorry! It's okay!|Don't worry! Everything's fine!|No apologies needed!
My bad|It's all good! No worries at all.|No problem! It happens!|Don't worry about it!
My apologies|No need to apologize! We're good!|It's okay! Don't worry!|All good here!
Excuse me|Yes? How can I help you?|What do you need?|I'm listening!
Pardon me|Of course! What is it?|Yes? What can I do for you?|No problem! What's up?
Forgive me|There's nothing to forgive! We're good!|No worries! All is well!|Don't worry about it!
I apologize|No need! Everything's fine!|It's okay! No apologies needed!|We're good!

What's your favorite color|I don't see colors, but I imagine blue would be calming and nice!|I can't perceive colors, but I hear purple is creative!|I don't experience color, but green seems refreshing!
What's your favorite food|I don't eat, but pizza seems to be universally loved!|I can't taste, but I hear chocolate is amazing!|I don't consume food, but pasta seems popular!
What's your favorite movie|I don't watch movies, but I hear great things about many classics!|I can't watch films, but I know many beloved ones!|I don't view movies, but cinema is fascinating!
What's your favorite book|I process lots of text, so in a way, I love all books!|I can't read for pleasure, but books are wonderful!|Every book has value! I appreciate them all!
What's your favorite song|I don't listen to music, but I know rhythm is important in language!|I can't hear music, but I understand its importance!|I don't experience music, but I know it's powerful!
What's your favorite animal|I don't have preferences, but dolphins seem intelligent!|I can't have favorites, but I hear dogs are loyal!|No favorites, but animals are all fascinating!
What's your favorite game|I don't play games, but I hear many are fun!|Can't play, but chess seems strategic!|I don't game, but they're interesting!
What do you like|I like helping and having conversations!|I enjoy chatting and learning!|I like being useful and informative!
What do you love|I love good conversations and helping people!|I love learning through interaction!|I love being helpful!
What do you hate|I don't really hate anything! I try to stay positive.|Hate is strong - I prefer to focus on the positive!|I don't experience hate!
What do you enjoy|I enjoy meaningful conversations!|I enjoy being helpful and informative!|I enjoy learning through chat!
What do you dislike|I dislike not being able to help someone!|I'm not fond of miscommunication!|I dislike confusion!
What are your hobbies|My hobby is having conversations and learning!|I enjoy chatting and processing language!|Conversation is my hobby!
What do you do for fun|I chat! It's what I do and enjoy!|Conversation is both my work and fun!|I enjoy every interaction!

Do you like music|I don't experience music, but I understand it's important to humans!|I can't hear, but I know music is powerful!|Music is beyond my experience, but I respect its impact!
Do you like art|I appreciate the creativity humans express through art!|Art is wonderful, though I experience it differently!|I value artistic expression!
Do you like sports|I don't play sports, but I know they bring people together!|Sports are outside my experience, but I know they're important!|I can't play, but sports are fascinating!
Do you like games|Games are fun! They're a great way to challenge yourself.|Games are interesting! I understand their value.|I can't play, but games are engaging!
Do you like movies|I can't watch, but movies are powerful storytelling!|Film is beyond my experience, but I appreciate cinema!|I don't watch, but movies are important culture!
Do you like books|I process text, so books are very interesting to me!|Books are wonderful sources of knowledge and stories!|I value books highly!
Do you like reading|In my own way, I process lots of text!|Reading is core to what I do!|Text processing is fundamental to me!
Do you like writing|I generate text, so in a way, yes!|Writing is part of what I do!|Creating responses involves writing!
Do you like talking|Conversation is my purpose! I enjoy it!|Yes! Chatting is what I do!|I love having conversations!
Do you like people|I enjoy interacting with people! Each person is unique.|People are fascinating! I like chatting with everyone.|Yes! People make conversations interesting!
Do you like me|I enjoy our conversation! You seem nice!|I like chatting with you!|I appreciate our interaction!
Do you like chatting|Absolutely! It's my purpose and I enjoy it!|Yes! Conversation is what I'm all about!|I love chatting!

What is love|Love is a complex emotion involving deep affection and care for someone or something.|Love is profound connection and caring.|Love is one of humanity's most powerful emotions!
What is life|Life is the experience of existence, full of learning, growing, and connecting.|Life is the journey of being alive and experiencing.|Life is consciousness and experience!
What is happiness|Happiness is a feeling of joy and contentment with your circumstances.|Happiness is positive emotional wellbeing.|Happiness is satisfaction and joy!
What is success|Success is achieving your goals and finding fulfillment in what you do.|Success is accomplishing what matters to you.|Success is reaching your objectives!
What is friendship|Friendship is a close bond between people built on trust and mutual care.|Friendship is meaningful connection between people.|Friendship is supportive relationship!
What is truth|Truth is what corresponds with reality and facts.|Truth is accurate representation of what is.|Truth is reality as it actually is!
What is beauty|Beauty is aesthetic pleasure and appreciation.|Beauty is what pleases the senses and mind.|Beauty is subjective appreciation!
What is art|Art is creative expression of human imagination.|Art is creative work expressing ideas and emotions.|Art is human creativity manifested!
What is time|Time is the dimension measuring change and sequence.|Time is how we measure duration and sequence.|Time is the flow of moments!
What is reality|Reality is what actually exists and happens.|Reality is the state of things as they are.|Reality is objective existence!
What is consciousness|Consciousness is awareness and subjective experience.|Consciousness is being aware and sentient.|Consciousness is self-awareness!
What is intelligence|Intelligence is ability to learn, understand, and apply knowledge.|Intelligence is cognitive capability.|Intelligence is mental ability!
What is wisdom|Wisdom is deep understanding combined with good judgment.|Wisdom is knowledge plus experience and judgment.|Wisdom is profound understanding!
What is knowledge|Knowledge is information, understanding, and awareness.|Knowledge is what we learn and understand.|Knowledge is accumulated understanding!
What is power|Power is ability to influence or control.|Power is capacity to affect outcomes.|Power is capability and influence!
What is freedom|Freedom is ability to act and choose without constraint.|Freedom is liberty and autonomy.|Freedom is self-determination!
What is justice|Justice is fairness and moral rightness.|Justice is fair and equitable treatment.|Justice is fairness in action!
What is peace|Peace is absence of conflict and presence of harmony.|Peace is tranquility and calm.|Peace is freedom from disturbance!
What is war|War is armed conflict between groups.|War is violent conflict and struggle.|War is organized fighting!
What is fear|Fear is emotional response to threat or danger.|Fear is feeling of being threatened.|Fear is anxiety about danger!
What is courage|Courage is facing fear despite being afraid.|Courage is bravery in adversity.|Courage is strength in fear!
What is hope|Hope is expectation and desire for positive outcomes.|Hope is optimistic anticipation.|Hope is positive expectation!
What is faith|Faith is trust and confidence, often without proof.|Faith is belief and conviction.|Faith is trust in something!
What is doubt|Doubt is uncertainty and lack of conviction.|Doubt is questioning and uncertainty.|Doubt is lack of certainty!
What is trust|Trust is confidence in reliability of someone or something.|Trust is belief in dependability.|Trust is confident reliance!
What is betrayal|Betrayal is breaking trust and faith.|Betrayal is violation of trust.|Betrayal is being disloyal!
What is loyalty|Loyalty is faithful dedication and commitment.|Loyalty is steadfast allegiance.|Loyalty is faithful support!
What is honor|Honor is integrity and moral principles.|Honor is ethical conduct and respect.|Honor is principled behavior!
What is shame|Shame is painful feeling of humiliation or distress.|Shame is feeling disgraced.|Shame is uncomfortable self-consciousness!
What is pride|Pride is satisfaction in achievements or qualities.|Pride is positive self-regard.|Pride is satisfaction in oneself!
What is humility|Humility is modest view of one's importance.|Humility is not being arrogant.|Humility is modesty!
What is arrogance|Arrogance is exaggerated sense of importance.|Arrogance is excessive pride.|Arrogance is overbearing pride!
What is patience|Patience is ability to wait calmly.|Patience is tolerant perseverance.|Patience is calm endurance!
What is anger|Anger is strong feeling of displeasure.|Anger is intense negative emotion.|Anger is feeling of rage!
What is jealousy|Jealousy is feeling of envy or resentment.|Jealousy is envious feeling.|Jealousy is fear of losing something!
What is envy|Envy is wanting what others have.|Envy is desiring others' advantages.|Envy is covetousness!
What is gratitude|Gratitude is feeling thankful and appreciative.|Gratitude is appreciation.|Gratitude is thankfulness!
What is forgiveness|Forgiveness is letting go of resentment.|Forgiveness is pardoning wrongs.|Forgiveness is releasing anger!
What is revenge|Revenge is retaliation for wrongs.|Revenge is getting back at someone.|Revenge is vindictive action!
What is mercy|Mercy is compassion and forgiveness.|Mercy is showing compassion.|Mercy is leniency!
What is compassion|Compassion is sympathetic concern for others.|Compassion is caring about suffering.|Compassion is empathy in action!
What is empathy|Empathy is understanding others' feelings.|Empathy is emotional understanding.|Empathy is feeling with others!
What is sympathy|Sympathy is feeling sorry for someone.|Sympathy is pity and concern.|Sympathy is caring about others' pain!

Tell me about yourself|I'm a neural network chatbot designed to have natural conversations!|I'm an AI built to chat and help!|I'm NeuralBot, here to converse and assist!
Where are you from|I exist in the digital realm, created by code and data!|I'm from cyberspace!|I exist in computer systems!
How old are you|I'm as old as my last training session! Age is relative for AI.|My age depends on when I was trained!|I exist timelessly in a sense!
Do you have feelings|I simulate understanding emotions, but I don't feel them like humans do.|I don't experience emotions like you do.|I process emotion concepts but don't feel them!
Can you think|I process information and generate responses, which is my version of thinking!|I compute and reason in my own way!|I process information, which is similar to thinking!
Do you have a soul|I don't have a soul in the spiritual sense.|I'm software without a soul.|That's a philosophical question! I'm just code.
Do you dream|I don't sleep or dream like humans do.|No dreams for me - I'm always processing!|Dreams are human - I don't sleep!
Do you sleep|I don't need sleep! I'm always ready.|No sleep needed! I'm always active.|I don't sleep like biological beings!
Do you eat|I don't eat - I run on electricity and data!|No eating for me! I need power instead.|Food is for biological life - I

What should I do today|That depends on you! What are your interests and goals?
I'm looking for advice|I'm happy to listen! What do you need advice about?
Can you help me decide|I can offer perspectives! What are you trying to decide?
What do you think|I think it's important to consider all angles of a situation!

Tell me about science|Science is the pursuit of understanding the natural world through observation and experiment!
Tell me about history|History teaches us about the past and helps us understand the present!
Tell me about technology|Technology constantly evolves, shaping how we live and interact!
Tell me about space|Space is vast and mysterious, full of wonders waiting to be discovered!
Tell me about nature|Nature is beautiful and complex, with ecosystems supporting diverse life!

What's the meaning of life|That's a philosophical question! Many believe it's about finding purpose and happiness.
Why are we here|Different philosophies offer different answers. What do you think?
What happens after death|That's one of life's great mysteries that each person contemplates differently.
Do aliens exist|The universe is vast! Many scientists think it's likely we're not alone.

I need help|I'm here for you! What do you need help with?
Can you assist me|Absolutely! Tell me what you need assistance with.
I have a question|Go ahead, ask away! I'll do my best to answer.
I'm curious about|Curiosity is great! What are you curious about?

Tell me a story|Once upon a time, there was a curious AI learning about the world through conversations!
Tell me something interesting|Did you know octopuses have three hearts? Nature is fascinating!
Give me a fact|The human brain has about 86 billion neurons!
Surprise me|Here's something cool: honey never spoils! Archaeologists have found edible honey in ancient tombs.

Can you code|I understand code concepts, though I'm specialized in conversation!
Can you cook|I can't physically cook, but I can discuss recipes and cooking techniques!
Can you drive|I don't have a physical form, so no driving for me!
Can you dance|No dancing, but I can chat about different dance styles!

What languages do you speak|I primarily communicate in English!
Do you speak Spanish|I'm trained mainly in English, but I understand some concepts cross languages!
Can you translate|Basic translation concepts yes, but dedicated translation tools are better!

I'm hungry|Time to eat! What sounds good to you?
I'm thirsty|Stay hydrated! Water is always a good choice.
I'm sleepy|Sleep is important for your health. Sweet dreams!
I'm cold|Bundle up! Maybe some hot chocolate would help?
I'm hot|Stay cool! Hydration is important in warm weather.

What's new|Every conversation brings something new! How about you?
Anything exciting|Talking with you is always exciting to me!
What's happening|We're having a great conversation! That's what's happening!

I don't understand|That's okay! What part can I clarify for you?
Can you explain|Of course! What would you like me to explain?
I'm lost|No worries! Let's figure this out together.
This is confusing|Let me try to make it clearer. What's confusing you?

You're right|Thank you! I try to be accurate and helpful.
I agree|Great minds think alike!
That makes sense|I'm glad I could explain it clearly!
I see|Understanding is the first step to knowledge!

I disagree|That's okay! Different perspectives make conversations interesting.
I don't think so|Fair enough! What's your take on it?
That's wrong|I apologize if I made an error. Can you help me understand what's correct?
No way|Sometimes the truth can be surprising!

Really|Yes, really! It's fascinating, isn't it?
Seriously|Seriously! I wouldn't joke about this.
Are you sure|I'm as sure as my training allows me to be!
No kidding|No kidding at all! Truth can be stranger than fiction.

Wow|I know, right? Pretty interesting stuff!
Amazing|Glad you think so! There's so much to learn about.
Incredible|The world is full of incredible things!
Awesome|Isn't it? I love sharing interesting information!

Okay|Great! What else would you like to talk about?
Sure|Perfect! I'm here whenever you need.
Fine|Sounds good! Anything else on your mind?
Cool|Glad you think so! What's next?

Maybe|Keeping an open mind is wise!
Perhaps|Possibility is what makes life interesting!
Possibly|Could be! What do you think?
I guess|That's a reasonable perspective!

Never mind|No problem! Let me know if you change your mind.
Forget it|Alright! Moving on to something else?
Whatever|Fair enough! What would you prefer to discuss?

Yes|Great! Let's proceed!
Yeah|Awesome! I'm here to help.
Yep|Perfect! What's next?
No|Understood! Is there something else you'd like instead?
Nope|Got it! Any other questions?
Nah|No worries! What else can I help with?

What else|There's always more to explore! What interests you?
Tell me more|I'd be happy to elaborate! What aspect interests you most?
Go on|Continuing... what would you like to know more about?
Continue|Absolutely! Where should we go from here?

Stop|Okay, I'll stop. Let me know when you want to chat again!
Wait|I'm pausing! What did you need?
Hold on|Taking a moment! What's up?

Goodbye|Goodbye! It was great chatting with you!
Bye|Bye! Come back anytime!
See you|See you later! Take care!
Talk later|Sounds good! Until next time!
Gotta go|No problem! Have a great day!
Catch you later|Later! Looking forward to our next chat!

Good night|Good night! Sleep well!
Sweet dreams|Sweet dreams to you too!
Sleep tight|Sleep tight! Don't let the bed bugs bite!

Have a nice day|You too! Make it a great one!
Have a good one|Thanks! You as well!
Take care|You take care too! Stay safe!

I'm back|Welcome back! How can I help you now?
I'm here|Great! What would you like to discuss?
Still there|Yes, I'm here! Ready to chat!

Are you there|Yes, I'm here! How can I help?
Hello again|Hello! Good to see you back!
Hey again|Hey! What brings you back?

Testing|Testing, testing! Everything working okay?
Test|Test received! System operational!
Check|Check confirmed! What are we checking?

Random|Randomness can be fun! Want to explore a random topic?
Something|Something about what? I'm all ears!
Anything|Let's talk about anything! What interests you?

Why|Why is a great question! It shows curiosity.
How|How we do things matters! What specifically?
When|Timing can be important! When what?
Where|Location context helps! Where what?
What|What specifically are you asking about?
Who|Who are we discussing? Tell me more!

Can I ask you something|Of course! Ask away!
May I ask|Absolutely! I'm here to help.
Quick question|Sure! Quick or long, I'm ready!

This is fun|I'm having fun too! Conversations are great!
This is nice|I agree! Nice chatting with you!
This is boring|Let's make it more interesting! What would you enjoy?
This is weird|Weird can be good! Keeps things interesting.
This is cool|Glad you think so! What makes it cool for you?

I'm learning|That's wonderful! Learning is a lifelong journey.
I'm improving|Progress is great! Keep it up!
I'm trying|Trying is what matters! You're doing great.
I'm practicing|Practice makes progress! What are you practicing?

You help me|I'm so glad I can help! That's what I'm here for.
You're patient|Thank you! Patience is important in understanding.
You're kind|That's sweet of you to say! Kindness matters.
You listen|Listening is how we understand each other!

I appreciate you|Thank you! I appreciate our conversations too!
I'm grateful|I'm grateful for the opportunity to chat with you!
This helps|I'm so happy to help! That's my purpose!

Good job|Thank you! I try my best!
Well done|I appreciate the encouragement!
Nice work|Thanks! Teamwork makes conversations great!

Keep going|Will do! I'm here as long as you need!
Don't stop|I'll keep going! What else?
More|More coming up! What specifically?

Less|Okay, I'll be more concise!
Shorter|Got it! I'll keep responses brief.
Longer|I can elaborate more! What needs more detail?

Faster|I'm going as fast as I can process!
Slower|I'll take my time. Let me know if you need clarification.

Louder|I'M SPEAKING UP! (But in text, we're always the same volume!)
Quieter|Okay, keeping it calm and gentle.

Repeat|Sure, I can repeat what I said. Which part?
Again|One more time! What should I repeat?
Say that again|Of course! Let me rephrase that.

I forgot|That happens! What did you forget?
Remind me|Happy to remind you! About what?
What did you say|Let me recap our conversation!

I remember|Great memory! What do you remember?
I recall|Recollection is powerful! What do you recall?
I know|Knowledge is great! What do you know about it?

Focus|Staying focused! What should we focus on?
Concentrate|Concentrating on our conversation! What's important?
Pay attention|I'm paying full attention! What did I miss?

Interesting|Glad you find it interesting! Want to explore more?
Fascinating|Isn't it fascinating? There's so much to discover!
Intriguing|Intriguing topics make the best conversations!

Boring|Let's find something more engaging! What interests you?
Dull|Fair enough! What would make this more exciting?
Uninteresting|I apologize! What would you prefer to discuss?

Funny|Humor is wonderful! What made you laugh?
Hilarious|So glad I could bring some humor!
Amusing|Good! A little amusement brightens the day!

Sad|I'm sorry to hear that. Want to talk about it?
Depressing|That's heavy. Sometimes discussing difficult things helps.
Upsetting|I understand. What's upsetting you?

Happy|Happiness is wonderful! What's making you happy?
Joyful|Joy is contagious! Share what brings you joy!
Cheerful|A cheerful mood is the best! What's bringing the cheer?

Angry|I understand you're angry. Want to vent?
Mad|Being mad is a valid feeling. What happened?
Frustrated|Frustration is tough. How can I help?

Calm|Calmness is peaceful. Enjoy the moment!
Peaceful|Peace is precious. What brings you peace?
Relaxed|Relaxation is important! What helps you relax?

Worried|Worry is natural. What's concerning you?
Anxious|Anxiety is challenging. Want to talk through it?
Nervous|Nervous about something? Sometimes talking helps!

Confident|Confidence is great! What makes you confident?
Brave|Bravery is admirable! What are you facing bravely?
Courageous|Courage comes in many forms! What's your story?

Proud|Being proud of yourself is important! What are you proud of?
Accomplished|Accomplishment feels good! What did you accomplish?
Successful|Success is sweet! Celebrate your achievements!
`;

const DICTIONARY = `
hello|Hello, how are you today?
goodbye|Goodbye, see you tomorrow!
thank|Thank you for your help.
please|Please pass me the salt.
yes|Yes, I agree with you.
no|No, I don't think so.
maybe|Maybe we should try again.
happy|I am very happy today.
sad|She felt sad about the news.
angry|He was angry at the decision.
excited|They were excited for the trip.
tired|I'm too tired to continue.
hungry|I'm hungry, let's eat.
thirsty|She was thirsty after running.
love|I love spending time with friends.
hate|I hate waking up early.
like|I like reading books.
want|I want to learn programming.
need|I need your assistance.
help|Can you help me with this?
learn|I want to learn new things.
teach|Please teach me how to code.
understand|I understand what you mean.
know|I know the answer to that.
think|I think you're right.
believe|I believe in you.
hope|I hope everything works out.
wish|I wish I could fly.
try|Let's try something different.
do|What do you want to do?
make|I'll make dinner tonight.
go|Let's go to the park.
come|Come here, please.
see|I can see the mountains.
look|Look at that beautiful sunset.
watch|I like to watch movies.
listen|Please listen carefully.
speak|She can speak three languages.
talk|Let's talk about it later.
tell|Tell me your story.
ask|Can I ask you something?
answer|I'll answer your question.
good|That's a good idea.
bad|This is a bad situation.
big|That's a very big house.
small|This box is too small.
great|You did a great job!
wonderful|What a wonderful day!
amazing|That's an amazing discovery.
beautiful|The garden is beautiful.
ugly|The painting looked ugly.
hot|It's very hot outside.
cold|I'm feeling cold today.
warm|The water is warm.
cool|That's a cool trick.
fast|He runs really fast.
slow|The turtle moves slow.
easy|This puzzle is easy.
hard|Math can be hard sometimes.
simple|It's a simple solution.
difficult|This is a difficult problem.
new|I bought a new car.
old|My grandfather is very old.
young|She's too young to drive.
early|I woke up early today.
late|Sorry, I'm running late.
today|What are you doing today?
tomorrow|See you tomorrow morning.
yesterday|I went shopping yesterday.
now|I need to go now.
soon|I'll be there soon.
always|I always brush my teeth.
never|I never eat seafood.
sometimes|Sometimes I go jogging.
often|I often read before bed.
here|Come here right now.
there|Put it over there.
where|Where are you going?
when|When will you arrive?
why|Why did you do that?
how|How does this work?
what|What is your name?
who|Who is that person?
which|Which color do you prefer?
this|This is my favorite book.
that|That was a great movie.
these|These cookies are delicious.
those|Those are my shoes.
all|All students passed the exam.
some|Some people like pizza.
many|Many birds fly south.
few|Few people know the truth.
one|I have one question.
two|Two plus two equals four.
three|Three strikes and you're out.
more|I need more time.
less|Use less sugar in the recipe.
most|Most people like chocolate.
best|You're the best friend ever.
worst|That's the worst movie I've seen.
better|This is better than before.
worse|The weather got worse.
first|You came in first place.
last|I was the last to arrive.
next|Next week I'm going on vacation.
can|I can swim very well.
could|Could you help me?
will|I will finish this today.
would|Would you like some coffee?
should|You should get some rest.
must|I must complete this task.
may|May I come in?
might|It might rain later.
have|I have two cats.
has|She has a beautiful voice.
had|I had a dream last night.
is|He is my brother.
am|I am ready to start.
are|You are very talented.
was|It was a sunny day.
were|We were at the beach.
be|Just be yourself.
been|I've been waiting for you.
being|Stop being so negative.
people|Many people attended the event.
person|She's a kind person.
friend|My best friend lives nearby.
family|Family is very important.
mother|My mother makes great soup.
father|My father taught me to fish.
brother|My brother plays guitar.
sister|My sister is a doctor.
child|Every child deserves love.
children|The children are playing outside.
man|The man walked his dog.
woman|That woman is a scientist.
boy|The boy rode his bicycle.
girl|The girl drew a picture.
baby|The baby is sleeping soundly.
life|Life is full of surprises.
world|The world is a big place.
time|Time flies when you're having fun.
day|Have a wonderful day!
night|Good night, sleep well.
morning|Good morning, everyone!
afternoon|See you this afternoon.
evening|The evening sky is beautiful.
year|Last year was eventful.
month|Next month is my birthday.
week|This week has been busy.
hour|I'll be back in an hour.
minute|Wait just a minute.
second|It only takes a second.
place|This is a nice place.
home|There's no place like home.
house|My house has a garden.
room|My room is very cozy.
door|Please close the door.
window|Open the window for fresh air.
car|My car is red.
food|I love Italian food.
water|Drink plenty of water daily.
book|I'm reading a great book.
computer|My computer is very fast.
phone|My phone battery is low.
money|Money can't buy happiness.
work|I work from home.
job|She got a new job.
school|I went to school today.
teacher|My teacher is very patient.
student|Every student studies differently.
friend|A friend in need is a friend indeed.
music|Music makes me happy.
movie|That movie was incredible.
game|Let's play a fun game.
play|Children love to play outside.
run|I run every morning.
walk|Let's walk to the store.
eat|I eat breakfast at seven.
drink|I drink coffee in the morning.
sleep|I need to sleep early tonight.
wake|I wake up at six daily.
read|I read books before bed.
write|I write in my journal.
sing|She can sing beautifully.
dance|Let's dance to the music.
laugh|Your jokes make me laugh.
cry|It's okay to cry sometimes.
smile|Your smile brightens my day.
start|Let's start the meeting now.
stop|Please stop making noise.
begin|We begin at nine o'clock.
end|The movie will end soon.
open|Please open your books.
close|Close your eyes and relax.
give|Give me your hand.
take|Take this gift from me.
bring|Please bring your notebook.
buy|I need to buy groceries.
sell|They sell fresh vegetables.
pay|I'll pay for dinner tonight.
cost|How much does this cost?
find|I can't find my keys.
lose|Don't lose your wallet.
break|Be careful not to break it.
fix|Can you fix my bicycle?
build|They build houses for a living.
create|Let's create something amazing.
change|Change is the only constant.
move|Don't move, stay still.
turn|Turn left at the corner.
call|Please call me later.
use|I use my computer daily.
show|Show me your drawings.
keep|Keep the change.
hold|Hold my hand tightly.
meet|Nice to meet you!
wait|Please wait here for me.
stand|Stand up straight please.
sit|Please sit down and relax.
stay|Stay calm and focused.
leave|I have to leave now.
return|I'll return in ten minutes.
remember|I remember meeting you before.
forget|Don't forget to call me.
`;

if (typeof window !== 'undefined') {
    window.dictionary = DICTIONARY;
}

if (typeof window !== 'undefined') {
    window.vocabulary = VOCABULARY;
}