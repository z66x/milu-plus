from common import *
cases = [("Answer: B",1),("The answer is C.",2),("Let's think... so **Answer: D**",3),("B",1),("(A) foo",0),
 ("Answer: 3",2),("the answer is a tricky one",None),("I think it depends",None),("Reasoning...\nAnswer: A\nB is wrong",0),
 ("Step 1 ... \nC",2),("",None)]
bad=0
for t,exp in cases:
    got,_=parse_answer(t,["x","y","z","w"])
    ok = got==exp; bad+= (not ok); print("OK " if ok else "BAD",repr(t),got,exp)
d=normalize({"question":"q","option1":"a","option2":"b","option3":"c","option4":"d","target":"option3","domain":"Law","subject":"s"},"Hindi",0)
assert d["gold"]==2, d
d["shots"]=[d]; print(loglik_prompt(d,1)); print(gen_prompt(d,True))
print("failures:",bad)
