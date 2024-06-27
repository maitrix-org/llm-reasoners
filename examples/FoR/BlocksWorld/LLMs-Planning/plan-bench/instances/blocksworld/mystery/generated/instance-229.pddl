(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i k h g)
(:init 
(harmony)
(planet i)
(planet k)
(planet h)
(planet g)
(province i)
(province k)
(province h)
(province g)
)
(:goal
(and
(craves i k)
(craves k h)
(craves h g)
)))