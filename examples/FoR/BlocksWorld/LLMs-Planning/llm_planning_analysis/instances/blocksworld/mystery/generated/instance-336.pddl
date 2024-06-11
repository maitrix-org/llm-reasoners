(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k h l)
(:init 
(harmony)
(planet k)
(planet h)
(planet l)
(province k)
(province h)
(province l)
)
(:goal
(and
(craves k h)
(craves h l)
)))