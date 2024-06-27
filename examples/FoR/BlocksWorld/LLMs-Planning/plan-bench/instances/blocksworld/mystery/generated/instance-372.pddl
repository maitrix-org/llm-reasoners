(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k a b)
(:init 
(harmony)
(planet k)
(planet a)
(planet b)
(province k)
(province a)
(province b)
)
(:goal
(and
(craves k a)
(craves a b)
)))