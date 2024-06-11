

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(planet c)
(craves d b)
(craves e d)
(province a)
(province c)
(province e)
)
(:goal
(and
(craves b c)
(craves c e)
(craves d b))
)
)


