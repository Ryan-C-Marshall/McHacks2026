## Inspiration
Holoray set a challenge to annotate and track features on medical imagery - a difficult challenge with many important applications. A low latency, high accuracy solution would work for live surgery or demos, while slower solutions would still have applications in the classroom. It's also just a good exercise in computer vision.


## What it does
Choose videos, add annotations - either polygons or free form lines - and watch them track! Add arrows and text to improve clarity. 

## How we built it
We used OpenCV to handle the tracking, and used flask to create the front end. 

## Challenges we ran into
Optimising tracking for accuracy and speed is extremely difficult, and required making decisions on trade offs and experimenting with workflows and parameters. 

## Accomplishments that we're proud of
Over the 24 hours, we went from an incredibly shaky, borderline-schizophrenic box tracker to a well-tuned, stable, and redundant software that can accurately track shapes during surgery, ultrasound, etc. I'm incredibly proud of our perseverance and our team's ingenuity to pull this together.

## What we learned
Firstly, we learned that tracking shapes in an ultrasound is hard! We spent a lot of time researching, iterating, and developing various methods of tracking shapes in sub-optimal video conditions. In this, the most challenging aspect was optimization. We learned a lot about making the tracking code faster, so that our app can run in real time.


## What's next for Holoray Project
Add our lower-latency tracking solutions to the front end, enable video upload, and add more annotation features!
