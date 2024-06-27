

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(craves c b)
(craves d c)
(planet e)
(province a)
(province d)
(province e)
)
(:goal
(and
(craves c e)
(craves d a))
)
)


