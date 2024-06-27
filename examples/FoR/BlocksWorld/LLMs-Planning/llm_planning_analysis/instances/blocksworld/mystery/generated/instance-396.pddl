(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i h a b)
(:init 
(harmony)
(planet i)
(planet h)
(planet a)
(planet b)
(province i)
(province h)
(province a)
(province b)
)
(:goal
(and
(craves i h)
(craves h a)
(craves a b)
)))