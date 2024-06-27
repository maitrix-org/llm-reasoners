

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a e)
(ontable b)
(on c b)
(on d a)
(ontable e)
(clear c)
(clear d)
)
(:goal
(and
(on b e)
(on c d)
(on d a))
)
)


