from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe
from deepeval import evaluate
from deepeval.dataset import Golden

task_completion = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4o",
    include_reason=True,
    verbose_mode=True
)

# The trip_planner_agent simulates the actual LLM agent
@observe(metrics=[task_completion])
def trip_planner_agent(input):
    destination = "Paris"
    days = 2

    @observe()
    def restaurant_finder(city):
        return ["Le Jules Verne", "Angelina Paris", "Septime"]

    @observe()
    def itinerary_generator(destination, days):
        return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

    itinerary = itinerary_generator(destination, days)
    restaurants = restaurant_finder(destination)

    output = []
    for i in range(days):
        output.append(f"{itinerary[i]} and eat at {restaurants[i]}")

    return ". ".join(output) + "."


# We are simulating function calling here
evaluate(observed_callback=trip_planner_agent, goldens=[Golden(input="Paris, 2")])

