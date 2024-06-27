(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e l a)
(:init 
(harmony)
(planet e)
(planet l)
(planet a)
(province e)
(province l)
(province a)
)
(:goal
(and
(craves e l)
(craves l a)
)))