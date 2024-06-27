(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i h g)
(:init 
(harmony)
(planet i)
(planet h)
(planet g)
(province i)
(province h)
(province g)
)
(:goal
(and
(craves i h)
(craves h g)
)))