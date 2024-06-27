(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k h i l)
(:init 
(harmony)
(planet k)
(planet h)
(planet i)
(planet l)
(province k)
(province h)
(province i)
(province l)
)
(:goal
(and
(craves k h)
(craves h i)
(craves i l)
)))