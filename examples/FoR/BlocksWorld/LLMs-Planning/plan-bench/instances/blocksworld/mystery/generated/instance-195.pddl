(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a h g)
(:init 
(harmony)
(planet a)
(planet h)
(planet g)
(province a)
(province h)
(province g)
)
(:goal
(and
(craves a h)
(craves h g)
)))