(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a b l)
(:init 
(harmony)
(planet a)
(planet b)
(planet l)
(province a)
(province b)
(province l)
)
(:goal
(and
(craves a b)
(craves b l)
)))