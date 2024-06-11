(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a g h k)
(:init 
(harmony)
(planet a)
(planet g)
(planet h)
(planet k)
(province a)
(province g)
(province h)
(province k)
)
(:goal
(and
(craves a g)
(craves g h)
(craves h k)
)))