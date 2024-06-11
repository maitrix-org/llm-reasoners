(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g h a k)
(:init 
(harmony)
(planet g)
(planet h)
(planet a)
(planet k)
(province g)
(province h)
(province a)
(province k)
)
(:goal
(and
(craves g h)
(craves h a)
(craves a k)
)))