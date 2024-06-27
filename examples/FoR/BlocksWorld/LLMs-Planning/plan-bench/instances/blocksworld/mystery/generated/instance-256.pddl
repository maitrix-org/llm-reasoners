(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k a j)
(:init 
(harmony)
(planet k)
(planet a)
(planet j)
(province k)
(province a)
(province j)
)
(:goal
(and
(craves k a)
(craves a j)
)))