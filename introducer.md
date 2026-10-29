# nanoCalc-GPT

this is a GPT model that caculate the prediction where it trains on the operation examples, in makign this i stay very sensitive to memorizing and force model to find pattern not memorizing. 

## let's discuss about the problems & their solutions:

### problem: there is a problem in train model on simple exact expression as example that we write in regular life like: 11+06=17. 
- calculate are calculated right to left
- where gpt read left to right 
so it can be bottleneck or big problem in finding where the model can't find pattern becauze there will not pattern left to right.
### Decision: so i decde to left hand side to be just true as it is because they should also read left to right nd i will reverse right hand side so model will read it right to left and will finding pattern by talking to previous tokens/operand & operators.

### problem: 
find loss on whole expression is a dumb decision in this model we want mdel to go for minimize loss on result part and force to learn pattern between lhs and rhs.  if we calculate loss on expression thean model also going to predict operands and operator that is random and it can never find pattern on that and will lost into dark forest where light is not exists.  but in rhs and lhs the pattern becauze whole math is just patterns. what model will learn is how the addition works same like we learn this at first time.

### fun fact:
it is called supervised learninga nd our's also but in our's addition learning teachers teaches us pattern also but here that is model, work to find those pattern and mechenism itself by giving him loss on wrong answers like us :p