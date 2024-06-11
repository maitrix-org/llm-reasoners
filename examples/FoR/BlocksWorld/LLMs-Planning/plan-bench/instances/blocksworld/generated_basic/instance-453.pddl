

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a d)
(on b c)
(ontable c)
(on d b)
(on e a)
(clear e)
)
(:goal
(and
(on b e)
(on d b))
)
)


