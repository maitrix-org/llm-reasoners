(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g k a)
(:init 
(harmony)
(planet g)
(planet k)
(planet a)
(province g)
(province k)
(province a)
)
(:goal
(and
(craves g k)
(craves k a)
)))