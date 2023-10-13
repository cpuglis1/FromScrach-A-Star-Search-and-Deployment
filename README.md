# A* Search From Scratch

Try the tool [here](https://gjecmasqfyerbwu3zkufjq.streamlit.app/)! 

## Background
Simply, most of AI can be understood as search. We create an agent to solve problems it did not know it would face. One approach that is sometimes suitable is State Space Search (SSS). In SSS we are turning a general problem into graph, and then finding a path through that graph. This begs the question, how to turn problem into graph?

## 4 Key Elements of Search
We can define a problem as a graph by framing it with the following characteristics.
  1) Set of States - includes relevant characteristics such as the initial, goal and failure state.
  2) Set of Actions - includes ways to change state.
  3) Transition Model - describes what happens when agent applies action in state.
  4) Cost Model - describes actions cost.

## A* Search
A* Search is an optimal and complete approach that expands fewest possible nodes to find the goal. It combines Uniform Cost Search (UCS) and Greedy Search and is defined as f(n) = g(n) + h(n), where g(n) is sum of actual prior costs (UCS) and h(n) is estimated cost to goal (Greedy Heuristic). f(n) is used as the key in the priority queue.

## From Scratch
This A* search was coded from scratch, unit tested, and thoroughly documented before implementation to share the learning!
