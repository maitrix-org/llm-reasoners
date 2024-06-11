(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects c g a d)
(:init 
(harmony)
(planet c)
(planet g)
(planet a)
(planet d)
(province c)
(province g)
(province a)
(province d)
)
(:goal
(and
(craves c g)
(craves g a)
(craves a d)
)))