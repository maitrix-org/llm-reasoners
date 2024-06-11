

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(planet b)
(craves c e)
(craves d b)
(craves e d)
(province a)
(province c)
)
(:goal
(and
(craves c a)
(craves d c)
(craves e d))
)
)


