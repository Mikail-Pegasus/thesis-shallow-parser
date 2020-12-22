import time
from collections import defaultdict
import nltk
import re
from nltk.classify.maxent import MaxentClassifier

nltk.config_megam('F:\Final Thesis data\megam_src\megam_0.92\megam.exe')

class MaxenPosTagger(nltk.TaggerI):
    def train(self, train_sents, algorithm='megam', rare_word_cutoff=1, rare_feat_cutoff=1, trace=3, **cutoffs):
        self.word_freqdist = self.gen_word_freqs(train_sents)
        self.featuresets = self.gen_featsets(train_sents,rare_word_cutoff)
        self.features_freqdist = self.gen_feat_freqs(self.featuresets)
        self.cutoff_rare_feats(self.featuresets, rare_feat_cutoff)
        start = time.time()
        self.classifier = MaxentClassifier.train(self.featuresets, algorithm, trace, **cutoffs)
        end = time.time()
        if trace > 0:
           print("Time to train the POS tagger: {0}".format(round(end - start, 3)))


    def gen_feat_freqs(self, featuresets):
        features_freqdist = defaultdict(int)
        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                features_freqdist[((feature, value), tag)] += 1
        return features_freqdist
    

    def gen_word_freqs(self, train_sents):
        word_freqdist = nltk.FreqDist()
        for tagged_sent in train_sents:
            for (word, _tag) in tagged_sent:
                word_freqdist[word] += 1
        return word_freqdist


    def gen_featsets(self, train_sents, rare_word_cutoff):
        featuresets = []
        for tagged_sent in train_sents:
            history = []
            untagged_sent = nltk.untag(tagged_sent)
            for (i, (_word, tag)) in enumerate(tagged_sent):
                featuresets.append((self.extract_features(untagged_sent, i, history, rare_word_cutoff), tag))
                history.append(tag)
        return featuresets

    def cutoff_rare_feats(self, featuresets, rare_feat_cutoff):
        never_cutoff_features = set(['w', 't'])
        for (feat_dict, tag) in featuresets:
            for (feature, value) in feat_dict.items():
                feat_value_tag = ((feature, value), tag)
                if self.features_freqdist[feat_value_tag] < rare_feat_cutoff:
                    if feature not in never_cutoff_features:
                        feat_dict.pop(feature)


    def extract_features(self, sentence, i, history, rare_word_cutoff=1):
        features = {}
        hyphen = re.compile("-")
        number = re.compile("\d")
        # generating features for: w-1, t-1, w-2, t-2 while taking care of the beginning of a sentence
        if i == 0:  # first word of sentence
            features.update({"w-1": "<Shuru>", "t-1": "<Shuru>",
                             "w-2": "<Shuru>", "t-2 t-1": "<Shuru> <Shuru>"})
        elif i == 1:  # second word of sentence
            features.update({"w-1": sentence[i - 1], "t-1": history[i - 1],
                             "w-2": "<Shuru>",
                             "t-2 t-1": "<Shuru> %s" % (history[i - 1])})
        else:
            features.update({"w-1": sentence[i - 1], "t-1": history[i - 1],
                             "w-2": sentence[i - 2],
                             "t-2 t-1": "%s %s" % (history[i - 2], history[i - 1])})

        # generating features for: w+1, w+2 while taking care of the end of a sentence.
        for inc in [1, 2]:
            try:
                features["w+%i" % (inc)] = sentence[i + inc]
            except IndexError:
                features["w+%i" % (inc)] = "<shesh>"

        if self.word_freqdist[sentence[i]] >= rare_word_cutoff:
            # additional features for 'non-rare' words
            features["w"] = sentence[i]

        else:  # additional features for 'rare' or 'unseen' words
            features.update({"suffix(1)": sentence[i][-1:], "suffix(2)": sentence[i][-2:],
                             "suffix(3)": sentence[i][-3:], "suffix(4)": sentence[i][-4:],
                             "prefix(1)": sentence[i][:1], "prefix(2)": sentence[i][:2],
                             "prefix(3)": sentence[i][:3], "prefix(4)": sentence[i][:4]})
            if hyphen.search(sentence[i]) != None:
                features["hyphen-exists"] = True
            if number.search(sentence[i]) != None:
                features["number-exists"] = True

        return features

    def tag(self, sentence, rare_word_cutoff=5):

        history = []
        for i in range(len(sentence)):
            featureset = self.extract_features(sentence, i, history, rare_word_cutoff)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        output = []
        for i in range(len(sentence)):
            output.append((sentence[i],history[i]))
        return output

def word_tokenizer(inp):
    w = inp.split()
    return w

lineList = [line.rstrip('\n') for line in open("bn_tagged_mod.txt", "r", encoding='utf-8')]
print(lineList)
sentence = [x.split() for x in lineList]
print(sentence)
tagged_sents = []
for x in sentence:
    b = [tuple(y.split('\\')) for y in x]
    tagged_sents.append(b)
print(tagged_sents)

size = int(len(tagged_sents)* 0.8)
train_sents, test_sents = tagged_sents[:size], tagged_sents[size:]
print(len(train_sents))
print(len(test_sents))
maxent_tagger = MaxenPosTagger()
maxent_tagger.train(train_sents)
print("Accuracy of the POS tagger: ", 100 * (maxent_tagger.evaluate(test_sents)), "%")
sent = input("Give input sentence :")
data = word_tokenizer(sent)
print(data)
print("Classified tagged format of unseen sentence: ", maxent_tagger.tag(data))
#print("Accuracy of the POS tagging: ", 100*(maxent_tagger.evaluate([maxent_tagger.tag(data)])),"%")
'''
inputs:

আমি তোমায় ভালোবাসি |
আমার সোনার বাংলা , আমি তোমায় ভালোবাসি |
সততা একটি মহৎ গুণ |
সে পরিশ্রমী |
রহিম ভাত খাচ্ছে |
রহিম সৎ তাই সে মহৎ |
সে নিজে কাজটা করেছে |
আমাকে যেতে দাও |
সে দ্রুত দৌড়াতে পারে |
ছিঃ , এমন কাজ তোর |
আহা ! লোকটি দেখতে পায় না |
আমি তার জন্য অপেক্ষা করছি |
তুমি ও আমি যাব |
সে কি আসবে ?
তার শরীরে ভিটামিন A - এর অভাব আছে |
খুব কম ছেলেই ঘটকের চোখে পাএী দেখে বিয়ে করতে চায় |
রপ্তানি দ্রব্য - তাজা ও শুকনা ফল , আফিম , পশুচর্ম ও পশম এবং কার্পেট |
রাজা মহানন্দ রাজধানীতে তৈরি করেছিল শিব মন্দির ও বৈষ্ণবদের মন্দির |
মলয়ের কাছে সব শুনে তার স্ত্রী বলল , " তোমার নাম কি মা ? "
নিজস্ব প্রতিনিধি |
অন্ধকার গাঢ়তর হয়ে ওঠে |
এর দ্বারা বশীকরণও হয় , আচার্য্যেরা এই বলে থাকেন |
বিশ্বাস , ঠিকও হয় |
আকবকের দরবারে যে ছত্রিশ জন বিশিষ্ট কলাকার ছিলেন , সকলেরই মতে , তাঁদের মধ্যে তানসেন ছিলেন সর্বশ্রেষ্ঠ |
সাবিত্রীদি চুড়ো বেঁধে দিচ্ছিলেন সতুর চুলে |
মনে হয় , হতাশা লাঘব করার চেষ্টা |
বুধবার এঁদের মধ্যে একজনের মৃত্যু হয় |
তাহার জন্য তাহাদের দাম্ভিকতা নাই |
মনে আছে , হেলিকপ্টারে চড়ে ঢুকেছিলাম জ্যান্ত আগ্নেয়গিরির গহ্বরে |
সশব্দে ট্রাক ছুটছে |
নিতান্ত গৃহবধূ ছিলেন না কোরাজন আকিনো |
তোমার বন্ধু বলল দু হাজার হবে |
রাণী লক্ষ্মীবাই রাজসভা আহ্বান করেছেন |
ইহা খাদ্যকে ধরিতে এবং য়্যানটিনা , প্যাল্প এবং অগ্রপদকে পরিষ্কার রাখিতে সাহায্য করে |
ভাইকে প্রাণাধিক স্নেহ করতেন রামকুমার |
নির্মাণের কাজ চলছে মন্থর গতিতে |
টেবিল অনুনাদী বস্তুর কাজ করে |
বাবার শেষকৃত্য সম্পন্ন করেই সীমা ফিরে এসেছিল কলকাতায় |
প্রদর্শনী খেলা দুটি - ব্যাডমিণ্টন ও বোলিং |
সাধারণত নেতাজী ডিনারটা আনন্দ করে খেতেন |
আয়নাটির নাম ভালবাসা |
যা বেরিয়ে যা |
// বিংশ শতাব্দীর বিশের দশকে উচ্চ বিভব সম্পন্ন প্রতি প্রভব বাতি ব্যবহার হত |
রাজ্য শাখার এক বিবৃতিতে বলা হয়েছে , সি পি আই ( এম ) দেশের একটি স্বীকৃত রাজনৈতিক দল |
এর ফলে মরু অঞ্চলে তীব্র নিম্নচাপ সৃষ্টি হয় |
এরই নাম মায়া !
মালদহ জেলার একটি জায়গা থেকে রাজা শশাঙ্কের একটি স্বর্ণমুদ্রাও তাঁরা সংগ্রহ করতে পেরেছেন |
আগামী বছর 1990 সালকে বিশ্ব সাক্ষরতা দিবস হিসাবে পালন করা হবে |
শিল্প শিল্পীর অন্তরেই প্রকাশিত |
বিহারীলালের বিবাহ হয় 19 বছর বয়সে |
অকুস্থল চীনের একটি শহর |
তার বেঁচে থাকাই কঠিন হয় |
1939 সালে ব্রিটেনে চার হাজার টিভি সেট বিক্রি হয় |
মোজায় নেই ফুটো |
চিত্রতারকার বুশশার্টখানা মদনা সন্ধানী চোখে দেখে নিয়েছিল |
চার বছরে পঞ্চম মুদ্রণ ভ্রমণ সাহিত্যে সত্যিই অভাবনীয় এবং আনন্দের |
আরও বড় কিছু , বেশী কিছু |
সারা হায়দ্রাবদ শহরে উত্সবের হাওয়া |
শীতকাল বড় প্রবঞ্চক |
দুজনেই ইঙ্গিতটা ধরতে পেরেছেন |
বল্লারশাহ এবং ওয়ারধা থেকে য়্যামবুলেনস নিয়ে ছুটে গিয়েছেন চিকিৎসকরা |
যাঁহারা ধীশক্তিসম্পন্ন , সূক্ষ্মদর্শী , চিন্তাশীল কবি ও রসিক , তাঁহারাই প্রবাদের সৃষ্টিকর্তা |
একমাত্র শ্রমই পণ্য - মূল্য সৃষ্টি করে এবং শ্রমই মূল্যের মাপকাঠি |
বলতো ওদের ফ্ল্যাটের দামী দামী জিনিস চুরি হয়ে গেছে তোমার দোষে |
দুদিন সবুর করো মশাই |
ঈগল পাখি মাছ খায় |
একটি ছেলে বই পড়ছে |
'''
#Chunking starts from here
grammar = """
          NPP: {<NP|NC>*<MD><AJ>*<NP|NC>*}
               {<PN>*<NP|NC>*<PN>*} 
          VPP: {<VM>*<VA>?} 
          """
cp = nltk.RegexpParser(grammar)
sss = []
for i in range(len(test_sents)):
    sss.append(cp.parse(test_sents[i]))
#print(cp.evaluate(sss))

result = cp.parse(maxent_tagger.tag(data))
print("IOB Chunked format of unseen sentence: ",nltk.chunk.tree2conlltags(result))
result.draw()
