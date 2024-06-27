(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l e a)
(:init 
(harmony)
(planet l)
(planet e)
(planet a)
(province l)
(province e)
(province a)
)
(:goal
(and
(craves l e)
(craves e a)
)))