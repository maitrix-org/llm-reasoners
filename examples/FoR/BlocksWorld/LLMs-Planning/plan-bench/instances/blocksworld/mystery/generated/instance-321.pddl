(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a g h c)
(:init 
(harmony)
(planet a)
(planet g)
(planet h)
(planet c)
(province a)
(province g)
(province h)
(province c)
)
(:goal
(and
(craves a g)
(craves g h)
(craves h c)
)))