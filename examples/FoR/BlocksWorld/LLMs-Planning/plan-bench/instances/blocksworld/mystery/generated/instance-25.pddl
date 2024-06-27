(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects j a c)
(:init 
(harmony)
(planet j)
(planet a)
(planet c)
(province j)
(province a)
(province c)
)
(:goal
(and
(craves j a)
(craves a c)
)))