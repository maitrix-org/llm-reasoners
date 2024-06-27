(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l h a b)
(:init 
(harmony)
(planet l)
(planet h)
(planet a)
(planet b)
(province l)
(province h)
(province a)
(province b)
)
(:goal
(and
(craves l h)
(craves h a)
(craves a b)
)))