

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a c)
(ontable b)
(on c d)
(on d b)
(ontable e)
(clear a)
(clear e)
)
(:goal
(and
(on a c)
(on d a)
(on e d))
)
)


